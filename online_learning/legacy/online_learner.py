#!/usr/bin/env python
"""
온라인 학습 Learner 프로세스.

CR-DAgger의 residual_learner_v1.py에 대응하되, 이 스택에는 residual policy가 없으므로
**base diffusion policy 자체를 낮은 LR로 online fine-tune** 하고 EMA 가중치를 actor로
hot-swap 시킨다. (relabel: 사람이 민 결과인 achieved pose가 action 타깃이 됨)

동작:
  1. base 체크포인트를 로드 (model, ema_model, optimizer, normalizer 포함)
  2. mailbox를 폴링해 actor가 보낸 새 correction 에피소드(HDF5)를 받는다
  3. 누적 HDF5(accumulated.hdf5)에 append
  4. 에피소드가 충분히 모이면 한 라운드 학습(EPOCHS_PER_ROUND) 후 EMA 가중치 발행
  5. 2로 반복

주의: diffusion policy는 CR-DAgger의 작은 residual MLP보다 학습이 무거워서
한 라운드가 수초~수분 걸린다. 이건 구조상 online이지만 갱신 주기는 그만큼 느리다.
(더 빠른 갱신이 필요하면 residual policy 변형이 필요 — README 참고)

실행:
  conda activate robodiff
  cd /home/rush/rush_diffusion_policy
  python online_learning/online_learner.py
"""
import os
import sys
import time
import warnings

# normalizer 계산 시 나오는 numpy "invalid value encountered in cast" 경고는
# 학습/추론에 영향 없는 알려진 무해 경고(README) — 로그 도배 방지용으로 무시.
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import h5py
import torch
import dill
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

from online_learning.legacy import config_online as C
from online_learning.legacy.mailbox import FileMailbox

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _copy_demos(src_hdf5, dst_data_group, start_idx):
    """src_hdf5의 data/demo_* 를 dst 그룹에 demo_{start_idx+i} 로 복사. 개수 반환."""
    n = 0
    with h5py.File(src_hdf5, "r") as sf:
        names = sorted(sf["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for name in names:
            sf.copy(f"data/{name}", dst_data_group, name=f"demo_{start_idx + n}")
            n += 1
    return n


class OnlineLearner:
    def __init__(self):
        self.mailbox = FileMailbox(C.ONLINE_WORKDIR)
        self.device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
        self.accumulated_hdf5 = os.path.join(C.ONLINE_WORKDIR, "accumulated.hdf5")
        self.num_demos = 0
        self.version = 0

        # base 체크포인트 로드 (model/ema_model/normalizer 포함)
        print(f"[Learner] base 체크포인트 로드: {C.BASE_CKPT}")
        payload = torch.load(open(C.BASE_CKPT, "rb"), pickle_module=dill, weights_only=False)
        self.cfg = payload["cfg"]
        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = cls(self.cfg)
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.model = self.workspace.model
        self.ema_model = self.workspace.ema_model if self.cfg.training.use_ema else None
        self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)

        # fine-tune용 optimizer (낮은 LR)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=C.LR,
                                           betas=(0.95, 0.999), weight_decay=1e-6)
        optimizer_to(self.optimizer, self.device)

        self.ema = None
        if self.ema_model is not None:
            self.ema = hydra.utils.instantiate(self.cfg.ema, model=self.ema_model)

        # 누적 HDF5 초기화 (+ 선택적으로 base demo 섞기)
        self._init_accumulated()

        # 시작 시 base(ema) 가중치를 v0로 발행해 actor가 즉시 쓸 수 있게 함
        self._publish()
        self._status("idle")

    def _status(self, state, **extra):
        s = {"state": state, "version": self.version - 1,
             "num_demos": self.num_demos}
        s.update(extra)
        self.mailbox.publish_status(s)

    def _init_accumulated(self):
        os.makedirs(C.ONLINE_WORKDIR, exist_ok=True)
        with h5py.File(self.accumulated_hdf5, "w") as f:
            data = f.create_group("data")
            if C.NUM_BASE_DEMOS_TO_MIX > 0 and os.path.exists(C.BASE_DATASET_PATH):
                with h5py.File(C.BASE_DATASET_PATH, "r") as bf:
                    names = sorted(bf["data"].keys(), key=lambda s: int(s.split("_")[1]))
                    for i, name in enumerate(names[:C.NUM_BASE_DEMOS_TO_MIX]):
                        bf.copy(f"data/{name}", data, name=f"demo_{i}")
                self.num_demos = min(C.NUM_BASE_DEMOS_TO_MIX, len(names))
                print(f"[Learner] base demo {self.num_demos}개를 누적셋에 mix")

    def add_episode(self, ep_hdf5):
        with h5py.File(self.accumulated_hdf5, "a") as f:
            n = _copy_demos(ep_hdf5, f["data"], self.num_demos)
        self.num_demos += n
        # 데이터가 바뀌었으니 stale zarr 캐시 제거
        for suff in (".zarr.zip", ".zarr.zip.lock"):
            p = self.accumulated_hdf5 + suff
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        print(f"[Learner] 에피소드 추가 (+{n}), 누적 demo={self.num_demos}")

    def _build_dataset(self):
        ds_cfg = OmegaConf.to_container(self.cfg.task.dataset, resolve=True)
        ds_cfg["dataset_path"] = self.accumulated_hdf5
        ds_cfg["use_cache"] = False   # 매 라운드 새 데이터 반영
        ds_cfg["val_ratio"] = 0.0
        target = ds_cfg.pop("_target_")
        cls = hydra.utils.get_class(target)
        return cls(**ds_cfg)

    def train_round(self):
        print(f"[Learner] ==== 학습 라운드 시작 (demo={self.num_demos}) ====")
        dataset = self._build_dataset()
        # 에피소드가 길면 매 epoch 샘플 수를 상한으로 잘라 라운드를 빠르게 끝냄
        sampler = None
        shuffle = True
        if C.MAX_SAMPLES_PER_EPOCH > 0 and len(dataset) > C.MAX_SAMPLES_PER_EPOCH:
            sampler = RandomSampler(dataset, replacement=True,
                                    num_samples=C.MAX_SAMPLES_PER_EPOCH)
            shuffle = False
            print(f"[Learner] epoch당 {C.MAX_SAMPLES_PER_EPOCH}/{len(dataset)} 샘플만 사용")
        loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=shuffle,
                            sampler=sampler, num_workers=C.NUM_WORKERS, pin_memory=True,
                            persistent_workers=False, drop_last=False)
        self.model.train()
        t0 = time.time()
        for epoch in range(C.EPOCHS_PER_ROUND):
            ep_loss = 0.0
            nb = 0
            for batch in loader:
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                loss = self.model.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.step(self.model)
                ep_loss += loss.item()
                nb += 1
            avg = ep_loss / max(nb, 1)
            self._status("training", epoch=epoch + 1, num_epochs=C.EPOCHS_PER_ROUND,
                         last_loss=avg)
            if epoch % 5 == 0 or epoch == C.EPOCHS_PER_ROUND - 1:
                print(f"[Learner]   epoch {epoch:3d}  loss={avg:.5f}")
        print(f"[Learner] 라운드 완료 ({time.time()-t0:.1f}s)")
        self._publish()
        self._status("published", last_loss=avg, round_time_s=round(time.time() - t0, 1))

    def _publish(self):
        """현재 EMA(없으면 model) 가중치를 mailbox로 발행."""
        src = self.ema_model if self.ema_model is not None else self.model
        state_dict = {k: v.detach().cpu() for k, v in src.state_dict().items()}
        self.mailbox.publish_weights({
            "state_dict": state_dict,
            "version": self.version,
            "num_demos": self.num_demos,
        })
        self.version += 1

    def run(self):
        print("[Learner] actor의 에피소드를 대기합니다 ...")
        while True:
            new_eps = self.mailbox.poll_new_episodes()
            if new_eps:
                self._status("received_episode")
                for ep in new_eps:
                    try:
                        self.add_episode(ep)
                    except Exception as e:
                        print(f"[Learner] 에피소드 로드 실패 {ep}: {e}")
                    finally:
                        self.mailbox.mark_episode_done(ep)
                if self.num_demos >= C.MIN_EPISODES_BEFORE_TRAIN:
                    self.train_round()
                else:
                    print(f"[Learner] 아직 학습 대기 (누적 {self.num_demos} < "
                          f"{C.MIN_EPISODES_BEFORE_TRAIN})")
            else:
                time.sleep(1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="온라인 학습 Learner. actor와 반드시 같은 base ckpt에서 시작해야 함.")
    parser.add_argument("--input", "-i", default=None,
                        help="base 체크포인트(.ckpt). 미지정 시 config_online.BASE_CKPT.")
    parser.add_argument("--base_dataset", default=None,
                        help="forgetting 완화용 base 학습 HDF5. 미지정 시 config 값.")
    args = parser.parse_args()
    if args.input is not None:
        C.BASE_CKPT = args.input
    if args.base_dataset is not None:
        C.BASE_DATASET_PATH = args.base_dataset
    print(f"[Learner] BASE_CKPT={C.BASE_CKPT}")
    print(f"[Learner] BASE_DATASET_PATH={C.BASE_DATASET_PATH} "
          f"(mix {C.NUM_BASE_DEMOS_TO_MIX} demos)")
    OnlineLearner().run()
