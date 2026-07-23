#!/usr/bin/env python
"""
온라인 residual DAgger Learner (cr-dagger residual_learner_v1.py 대응).

full-finetune 경로의 online_learning/finetune_teleop_learner.py 와 구조는 같지만:
  * base diffusion 을 fine-tune 하지 않는다. **frozen slow base + 작은 residual head** 만 학습.
    -> 망각(#2) 원천 차단, 라운드 <1s~수초.
  * 매 라운드 accumulated 교정 데이터로 FastResidualContextStepDataset 을 만들고,
    warm-continue(optimizer/epoch 유지) 로 head 만 학습.
  * 발행 가중치 = head(+선택 force_encoder) state_dict + normalizer state_dict.
    (frozen slow ~1.5GB 는 안 보냄. actor 는 같은 slow ckpt 를 이미 갖고 있음.)

이 learner 는 teleop 판(제어를 servo/manus 로 넘기는 개입)과 intervention 판(제어를 안
넘기고 사람이 팔을 미는 개입)이 **공유**한다. 차이는 (a) 어떤 config 모듈(self.C)을 쓰는가,
(b) `_make_sampler` 훅뿐이다. intervention 판은 residual_intervention_learner.py 가
이 클래스를 상속해 config 를 config_residual_intervention 으로, `_make_sampler` 를 개입
프레임 가중 샘플러로 override 한다.

데이터 흐름:
  actor --(residual 포맷 에피소드 HDF5)--> mailbox.transitions
  learner: accumulated.hdf5 에 append -> head warm-continue 학습 -> head 가중치 발행
  actor: 다음 에피소드 시작 시 head hot-swap

실행 (env 는 timm 있는 bae_robodiff):
  export RESIDUAL_ONLINE_WORKDIR=data/online_runs/run_hand_residual
  /home/rush/anaconda3/envs/bae_robodiff/bin/python online_learning/residual_teleop_learner.py
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
    # 로그 prefix (서브클래스가 override 해 터미널 구분: teleop=[R-Learner] / intervention=[I-Learner])
    tag = "[R-Learner]"

    def __init__(self, cfg=None):
        # cfg: config 모듈(속성명은 config_residual_online 과 동일해야 함). None 이면 teleop 판.
        self.C = cfg if cfg is not None else C
        self.mailbox = FileMailbox(self.C.ONLINE_WORKDIR)
        self.device = torch.device(self.C.DEVICE if torch.cuda.is_available() else "cpu")
        self.accumulated_hdf5 = os.path.join(self.C.ONLINE_WORKDIR, "accumulated.hdf5")
        self.num_demos = 0
        self.version = 0
        self.epoch = 0
        self.global_step = 0

        # residual 정책 설정 compose (slow_ckpt override)
        config_dir = os.path.join(ROOT, "diffusion_policy", "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            self.cfg = compose(config_name=self.C.RESIDUAL_CONFIG_NAME)
        # slow_ckpt 경로에 '='(epoch=0500)가 있어 hydra override 파서를 못 쓴다.
        # compose 후 직접 대입하고 resolve 하면 ${task.slow_ckpt_path} 보간이 반영된다.
        self.cfg.task.slow_ckpt_path = self.C.SLOW_CKPT
        OmegaConf.resolve(self.cfg)

        print(f"{self.tag} slow(base) ckpt 로 residual 정책 인스턴스화: {self.C.SLOW_CKPT}")
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        self.policy.to(self.device)

        self.trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in self.trainable_params)
        print(f"{self.tag} 학습 파라미터 수: {n_train:,} (head{'+force_encoder' if self.policy.train_force_encoder else ''})")
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=self.C.LR, betas=(0.95, 0.999), weight_decay=1e-6)
        optimizer_to(self.optimizer, self.device)

        self._init_accumulated()
        self._status("idle")

    # ── 상태 ────────────────────────────────────────────────────────────────
    def _status(self, state, **extra):
        s = {"state": state, "version": self.version - 1, "num_demos": self.num_demos}
        s.update(extra)
        self.mailbox.publish_status(s)

    def _init_accumulated(self):
        os.makedirs(self.C.ONLINE_WORKDIR, exist_ok=True)
        with h5py.File(self.accumulated_hdf5, "w") as f:
            f.create_group("data")

    # ── 에피소드 수신 ─────────────────────────────────────────────────────────
    def add_episode(self, ep_hdf5):
        with h5py.File(self.accumulated_hdf5, "a") as f:
            n = _copy_demos(ep_hdf5, f["data"], self.num_demos)
        self.num_demos += n
        print(f"{self.tag} 에피소드 추가 (+{n}), 누적 demo={self.num_demos}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    def _build_dataset(self, overrides=None):
        """overrides: 데이터셋 cfg 를 덮어쓸 항목(예: 개입 판의 action_key 교체).
        서브클래스가 데이터셋을 두 번 만들지 않고 한 번에 원하는 타깃으로 짓게 하는 훅."""
        ds_cfg = OmegaConf.to_container(self.cfg.task.dataset, resolve=True)
        ds_cfg["dataset_path"] = self.accumulated_hdf5
        ds_cfg["val_ratio"] = 0.0
        if overrides:
            ds_cfg.update(overrides)
        target = ds_cfg.pop("_target_")
        cls = hydra.utils.get_class(target)
        sig = inspect.signature(cls.__init__)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            accepted = set(sig.parameters) - {"self"}
            ds_cfg = {k: v for k, v in ds_cfg.items() if k in accepted}
        return cls(**ds_cfg)

    # ── 샘플러 훅 (서브클래스가 override) ────────────────────────────────────────
    def _make_sampler(self, dataset):
        """(sampler, shuffle) 반환. 기본: MAX_SAMPLES 상한이 있으면 RandomSampler(복원추출),
        아니면 sampler=None + shuffle=True. intervention 판은 개입 프레임 가중 샘플러로 override."""
        if self.C.MAX_SAMPLES_PER_EPOCH > 0 and len(dataset) > self.C.MAX_SAMPLES_PER_EPOCH:
            return RandomSampler(
                dataset, replacement=True, num_samples=self.C.MAX_SAMPLES_PER_EPOCH), False
        return None, True

    # ── 학습 라운드 (warm-continue) ────────────────────────────────────────────
    def train_round(self):
        dataset = self._build_dataset()
        if len(dataset) == 0:
            print(f"{self.tag} 유효 샘플 0개 — 스킵")
            return
        normalizer = dataset.get_normalizer()
        self.policy.set_normalizer(normalizer)
        self.policy.normalizer.to(self.device)

        sampler, shuffle = self._make_sampler(dataset)
        loader = DataLoader(
            dataset, batch_size=self.C.BATCH_SIZE, shuffle=shuffle, sampler=sampler,
            num_workers=self.C.NUM_WORKERS, pin_memory=True, drop_last=False)

        num_epochs = self.C.FIRST_EPOCHS if self.global_step == 0 else self.C.EPOCHS_PER_ROUND
        print(f"{self.tag} ==== 학습 라운드 (demo={self.num_demos}, samples={len(dataset)}, "
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
                print(f"{self.tag}   epoch {e:3d}  loss={avg:.6f}")
            self._status("training", epoch=e + 1, num_epochs=num_epochs, last_loss=avg)
        print(f"{self.tag} 라운드 완료 ({time.time() - t0:.1f}s), loss={avg:.6f}")
        self._publish()
        self._status("published", last_loss=avg, round_time_s=round(time.time() - t0, 1))

    # ── 가중치 발행 (head + normalizer 만) ──────────────────────────────────────
    def _extra_payload(self):
        """발행 payload 에 덧붙일 항목(서브클래스 훅). 개입형은 지연 모델(α)을 실어
        actor 가 α⁻¹ 게인으로 쓰게 한다."""
        return {}

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
        payload.update(self._extra_payload())
        self.mailbox.publish_weights(payload)
        self.version += 1

    # ── 대배치 DAgger 게이팅 (원본 CR-DAgger) ─────────────────────────────────────
    def _should_train(self):
        """원본 CR-DAgger 방식의 대배치 업데이트 게이팅.
        · 첫 학습: FIRST_TRAIN_EPISODES 개 모일 때까지 대기(num_episodes_before_first_training).
        · 이후: UPDATE_EVERY_N_EPISODES 개 새로 쌓일 때마다 갱신.
        논문: 소배치 잦은 갱신은 불안정(catastrophic forgetting) → 모아서 갱신."""
        first_n = int(getattr(self.C, "FIRST_TRAIN_EPISODES",
                              getattr(self.C, "MIN_EPISODES_BEFORE_TRAIN", 1)))
        every_n = max(1, int(getattr(self.C, "UPDATE_EVERY_N_EPISODES", 1)))
        if not self._trained_once:
            if self.num_demos >= first_n:
                return True
            print(f"{self.tag} 첫 학습 대기 (누적 {self.num_demos} < FIRST_TRAIN={first_n})")
            return False
        new_since = self.num_demos - self._demos_at_last_train
        if new_since >= every_n:
            return True
        print(f"{self.tag} 업데이트 대기 (신규 {new_since} < EVERY_N={every_n})")
        return False

    # ── restart-safe 버퍼 복원 (workflow B: 수집 후/재시작 실행) ───────────────────
    def _rebuild_from_disk(self):
        """transitions/ 의 모든 에피소드(.done 포함)로 accumulated 버퍼와 num_demos 를 복원.
        learner 를 껐다 켜도(수집 먼저→학습 나중, 배치마다 재시작) 이전 데이터를 잃지 않는다.
        __init__ 이 accumulated 를 비운 직후 호출된다."""
        eps = self.mailbox.list_all_episodes()
        if not eps:
            return
        for ep in eps:
            try:
                self.add_episode(ep)
                self.mailbox.mark_episode_done(ep)   # .ready 였으면 done 처리(poll 재수집 방지)
            except Exception as e:
                print(f"{self.tag} 재수집 실패 {ep}: {e}")
        print(f"{self.tag} 디스크에서 {self.num_demos}개 에피소드 버퍼 복원(restart-safe)")

    # ── 메인 루프 ──────────────────────────────────────────────────────────────
    def run(self):
        # restart-safe: 디스크의 전체 에피소드로 버퍼 복원 (workflow B).
        self._rebuild_from_disk()
        # 가중치 발행: 콜드 스타트만 빈 v0 발행(actor 부트스트랩 → slow-only).
        # 이미 발행된 head 가 있으면(재시작) 그대로 두고 버전만 이어받아, 재학습 완료 전까지
        # 기존(학습된) head 를 유지한다 — 빈 head 로 덮어써 actor 를 퇴보시키지 않기 위함.
        latest = self.mailbox.get_latest_weight_version()
        if latest is None:
            self._publish()
        else:
            self.version = latest + 1
            print(f"{self.tag} 재시작: 기존 head v{latest} 유지 → 다음 발행 v{self.version}")
        self._trained_once = False
        self._demos_at_last_train = 0
        print(f"{self.tag} actor 의 에피소드를 대기합니다 (첫 학습까지 "
              f"{getattr(self.C, 'FIRST_TRAIN_EPISODES', self.C.MIN_EPISODES_BEFORE_TRAIN)}개, "
              f"이후 {getattr(self.C, 'UPDATE_EVERY_N_EPISODES', 1)}개마다) ...")
        # 재시작 시 이미 충분히 쌓여 있으면(workflow B) 즉시 한 번 학습.
        if self._should_train():
            self.train_round()
            self._trained_once = True
            self._demos_at_last_train = self.num_demos
        while True:
            new_eps = self.mailbox.poll_new_episodes()
            if new_eps:
                self._status("received_episode")
                for ep in new_eps:
                    try:
                        self.add_episode(ep)
                    except Exception as e:
                        print(f"{self.tag} 에피소드 로드 실패 {ep}: {e}")
                    finally:
                        self.mailbox.mark_episode_done(ep)
                if self._should_train():
                    self.train_round()
                    self._trained_once = True
                    self._demos_at_last_train = self.num_demos
            else:
                time.sleep(1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="온라인 residual DAgger learner (teleop 판).")
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
