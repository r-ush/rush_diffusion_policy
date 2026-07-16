#!/usr/bin/env python
"""
온라인 residual DAgger Learner (cr-dagger residual_learner_v1.py 대응).

full-finetune 경로의 online_learning/online_learner.py 와 구조는 같지만:
  * base diffusion 을 fine-tune 하지 않는다. **frozen slow base + 작은 residual head** 만 학습.
    -> 망각(#2) 원천 차단, 라운드 <1s~수초.
  * 매 라운드 accumulated 교정 데이터로 FastResidualContextStepDataset 을 만들고,
    warm-continue(optimizer/epoch 유지) 로 head 만 학습.
  * 발행 가중치 = head(+선택 force_encoder) state_dict + normalizer state_dict.
    (frozen slow ~1.5GB 는 안 보냄. actor 는 같은 slow ckpt 를 이미 갖고 있음.)

데이터 흐름:
  actor --(residual 포맷 에피소드 HDF5)--> mailbox.transitions
  learner: accumulated.hdf5 에 append -> head warm-continue 학습 -> head 가중치 발행
  actor: 다음 에피소드 시작 시 head hot-swap

실행 (env 는 timm 있는 bae_robodiff):
  export RESIDUAL_ONLINE_WORKDIR=data/online_runs/run_hand_residual
  /home/rush/anaconda3/envs/bae_robodiff/bin/python online_learning/residual_online_learner.py
"""
import os
import sys
import time
import inspect
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in cast")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
import torch
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

from online_learning import config_residual_online as C
from online_learning.mailbox import FileMailbox

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _copy_demos(src_hdf5, dst_data_group, start_idx):
    """src_hdf5 의 data/demo_* 를 dst 그룹에 demo_{start_idx+i} 로 복사. 개수 반환."""
    n = 0
    with h5py.File(src_hdf5, "r") as sf:
        names = sorted(sf["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for name in names:
            sf.copy(f"data/{name}", dst_data_group, name=f"demo_{start_idx + n}")
            n += 1
    return n


class ResidualOnlineLearner:
    def __init__(self):
        self.mailbox = FileMailbox(C.ONLINE_WORKDIR)
        self.device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
        self.accumulated_hdf5 = os.path.join(C.ONLINE_WORKDIR, "accumulated.hdf5")
        self.num_demos = 0
        self.version = 0
        self.epoch = 0
        self.global_step = 0

        # residual 정책 설정 compose (slow_ckpt override)
        config_dir = os.path.join(ROOT, "diffusion_policy", "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            self.cfg = compose(config_name=C.RESIDUAL_CONFIG_NAME)
        # slow_ckpt 경로에 '='(epoch=0500)가 있어 hydra override 파서를 못 쓴다.
        # compose 후 직접 대입하고 resolve 하면 ${task.slow_ckpt_path} 보간이 반영된다.
        self.cfg.task.slow_ckpt_path = C.SLOW_CKPT
        OmegaConf.resolve(self.cfg)

        print(f"[R-Learner] slow(base) ckpt 로 residual 정책 인스턴스화: {C.SLOW_CKPT}")
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        self.policy.to(self.device)

        self.trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in self.trainable_params)
        print(f"[R-Learner] 학습 파라미터 수: {n_train:,} (head{'+force_encoder' if self.policy.train_force_encoder else ''})")
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=C.LR, betas=(0.95, 0.999), weight_decay=1e-6)
        optimizer_to(self.optimizer, self.device)

        self._init_accumulated()
        self._status("idle")

    # ── 상태 ────────────────────────────────────────────────────────────────
    def _status(self, state, **extra):
        s = {"state": state, "version": self.version - 1, "num_demos": self.num_demos}
        s.update(extra)
        self.mailbox.publish_status(s)

    def _init_accumulated(self):
        os.makedirs(C.ONLINE_WORKDIR, exist_ok=True)
        with h5py.File(self.accumulated_hdf5, "w") as f:
            f.create_group("data")

    # ── 에피소드 수신 ─────────────────────────────────────────────────────────
    def add_episode(self, ep_hdf5):
        with h5py.File(self.accumulated_hdf5, "a") as f:
            n = _copy_demos(ep_hdf5, f["data"], self.num_demos)
        self.num_demos += n
        print(f"[R-Learner] 에피소드 추가 (+{n}), 누적 demo={self.num_demos}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    def _build_dataset(self):
        ds_cfg = OmegaConf.to_container(self.cfg.task.dataset, resolve=True)
        ds_cfg["dataset_path"] = self.accumulated_hdf5
        ds_cfg["val_ratio"] = 0.0
        target = ds_cfg.pop("_target_")
        cls = hydra.utils.get_class(target)
        sig = inspect.signature(cls.__init__)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            accepted = set(sig.parameters) - {"self"}
            ds_cfg = {k: v for k, v in ds_cfg.items() if k in accepted}
        return cls(**ds_cfg)

    # ── 학습 라운드 (warm-continue) ────────────────────────────────────────────
    def train_round(self):
        dataset = self._build_dataset()
        if len(dataset) == 0:
            print("[R-Learner] 유효 샘플 0개 — 스킵")
            return
        normalizer = dataset.get_normalizer()
        self.policy.set_normalizer(normalizer)
        self.policy.normalizer.to(self.device)

        sampler = None
        shuffle = True
        if C.MAX_SAMPLES_PER_EPOCH > 0 and len(dataset) > C.MAX_SAMPLES_PER_EPOCH:
            sampler = RandomSampler(dataset, replacement=True, num_samples=C.MAX_SAMPLES_PER_EPOCH)
            shuffle = False
        loader = DataLoader(
            dataset, batch_size=C.BATCH_SIZE, shuffle=shuffle, sampler=sampler,
            num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=False)

        num_epochs = C.FIRST_EPOCHS if self.global_step == 0 else C.EPOCHS_PER_ROUND
        print(f"[R-Learner] ==== 학습 라운드 (demo={self.num_demos}, samples={len(dataset)}, "
              f"epochs={num_epochs}) ====")
        self.policy.train()
        t0 = time.time()
        avg = float("nan")
        for e in range(num_epochs):
            losses = []
            for batch in loader:
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                loss = self.policy.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
                self.global_step += 1
            avg = float(np.mean(losses)) if losses else float("nan")
            self.epoch += 1
            if e % 10 == 0 or e == num_epochs - 1:
                print(f"[R-Learner]   epoch {e:3d}  loss={avg:.6f}")
            self._status("training", epoch=e + 1, num_epochs=num_epochs, last_loss=avg)
        print(f"[R-Learner] 라운드 완료 ({time.time() - t0:.1f}s), loss={avg:.6f}")
        self._publish()
        self._status("published", last_loss=avg, round_time_s=round(time.time() - t0, 1))

    # ── 가중치 발행 (head + normalizer 만) ──────────────────────────────────────
    def _publish(self):
        payload = {
            "version": self.version,
            "num_demos": self.num_demos,
            "head_state": {k: v.detach().cpu() for k, v in self.policy.head.state_dict().items()},
            "normalizer_state": {k: v.detach().cpu() for k, v in self.policy.normalizer.state_dict().items()},
            "force_encoder_state": None,
        }
        if self.policy.train_force_encoder and self.policy.force_encoder is not None:
            payload["force_encoder_state"] = {
                k: v.detach().cpu() for k, v in self.policy.force_encoder.state_dict().items()}
        self.mailbox.publish_weights(payload)
        self.version += 1

    # ── 메인 루프 ──────────────────────────────────────────────────────────────
    def run(self):
        # 시작 시 (아직 학습 전) head 를 v0 로 발행해 actor 가 즉시 slow+random-residual 로 시작
        self._publish()
        print("[R-Learner] actor 의 에피소드를 대기합니다 ...")
        while True:
            new_eps = self.mailbox.poll_new_episodes()
            if new_eps:
                self._status("received_episode")
                for ep in new_eps:
                    try:
                        self.add_episode(ep)
                    except Exception as e:
                        print(f"[R-Learner] 에피소드 로드 실패 {ep}: {e}")
                    finally:
                        self.mailbox.mark_episode_done(ep)
                if self.num_demos >= C.MIN_EPISODES_BEFORE_TRAIN:
                    self.train_round()
                else:
                    print(f"[R-Learner] 학습 대기 (누적 {self.num_demos} < {C.MIN_EPISODES_BEFORE_TRAIN})")
            else:
                time.sleep(1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="온라인 residual DAgger learner.")
    parser.add_argument("--slow_ckpt", default=None, help="frozen slow base ckpt. 미지정 시 config 값.")
    parser.add_argument("--config_name", default=None, help="residual hydra config 이름.")
    args = parser.parse_args()
    if args.slow_ckpt is not None:
        C.SLOW_CKPT = args.slow_ckpt
    if args.config_name is not None:
        C.RESIDUAL_CONFIG_NAME = args.config_name
    print(f"[R-Learner] SLOW_CKPT={C.SLOW_CKPT}")
    print(f"[R-Learner] CONFIG={C.RESIDUAL_CONFIG_NAME}  WORKDIR={C.ONLINE_WORKDIR}")
    ResidualOnlineLearner().run()
