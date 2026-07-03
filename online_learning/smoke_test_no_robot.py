#!/usr/bin/env python
"""
로봇 없이 온라인 학습 루프 전체를 검증하는 스모크 테스트.

실제 로봇 대신, 학습 데이터셋의 몇 개 demo를 "actor가 보낸 correction 에피소드"인 것처럼
mailbox에 넣고, learner를 몇 라운드 돌린 뒤:
  - 가중치가 발행되는지 (version 증가)
  - 발행된 가중치를 actor 입장에서 로드해 policy에 hot-swap 하고 predict_action이 되는지
를 확인한다.

빠르게 끝나도록 config를 임시로 축소(EPOCHS_PER_ROUND 작게, demo 1개)해서 돈다.

실행:
  conda activate robodiff
  cd /home/rush/rush_diffusion_policy
  python online_learning/smoke_test_no_robot.py
"""
import os
import sys
import shutil
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
import torch

from online_learning import config_online as C


def make_fake_episode_from_dataset(dataset_path, demo_idx, out_path, max_len=40):
    """데이터셋의 한 demo를 그대로 1-demo HDF5(actor 전송 포맷)로 복사 (길이 제한)."""
    with h5py.File(dataset_path, "r") as f:
        d = f[f"data/demo_{demo_idx}"]
        img = np.array(d["obs/image0"])[:max_len]
        pose = np.array(d["obs/robot_pose_L"])[:max_len]
        quat = np.array(d["obs/robot_quat_L"])[:max_len]
        act = np.array(d["actions"])[:max_len]
    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        g = data.create_group("demo_0")
        o = g.create_group("obs")
        o.create_dataset("robot_pose_L", data=pose.astype(np.float32))
        o.create_dataset("robot_quat_L", data=quat.astype(np.float32))
        o.create_dataset("image0", data=img.astype(np.uint8))
        g.create_dataset("actions", data=act.astype(np.float32))
    return out_path


def main():
    # ── config를 스모크 테스트용으로 임시 축소 ──
    C.ONLINE_WORKDIR = tempfile.mkdtemp(prefix="online_smoke_")
    C.EPOCHS_PER_ROUND = 2
    C.BATCH_SIZE = 4
    C.NUM_WORKERS = 0
    C.MIN_EPISODES_BEFORE_TRAIN = 1
    C.NUM_BASE_DEMOS_TO_MIX = 0
    print(f"[smoke] workdir = {C.ONLINE_WORKDIR}")

    # learner를 import (config를 먼저 바꾼 뒤 import 되도록 지연 import)
    from online_learning.online_learner import OnlineLearner
    from online_learning.mailbox import FileMailbox

    learner = OnlineLearner()
    assert learner.mailbox.get_latest_weight_version() == 0, "초기 v0 발행 실패"
    print("[smoke] 초기 가중치 v0 발행 확인 OK")

    # 가짜 에피소드 2개를 mailbox transitions에 투입
    mb = FileMailbox(C.ONLINE_WORKDIR)
    for i, demo_idx in enumerate([0, 1]):
        ep = make_fake_episode_from_dataset(
            C.BASE_DATASET_PATH, demo_idx,
            os.path.join(C.ONLINE_WORKDIR, f"fake_ep_{i}.hdf5"))
        mb.send_episode(ep)

    # learner가 에피소드를 받아 한 라운드 학습하도록 poll->train 1회 수동 구동
    new_eps = learner.mailbox.poll_new_episodes()
    assert len(new_eps) == 2, f"에피소드 폴링 실패: {new_eps}"
    for ep in new_eps:
        learner.add_episode(ep)
        learner.mailbox.mark_episode_done(ep)
    assert learner.num_demos == 2
    v_before = learner.version
    learner.train_round()
    v_after = learner.version
    assert v_after == v_before + 1, "학습 후 가중치 버전 증가 실패"
    latest = learner.mailbox.get_latest_weight_version()
    print(f"[smoke] 학습 라운드 후 최신 가중치 버전 = {latest}")

    # ── actor 입장: 최신 가중치를 base policy에 hot-swap 후 predict_action ──
    import dill, hydra
    from omegaconf import OmegaConf
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
    from diffusion_policy.common.pytorch_util import dict_apply
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    payload = torch.load(open(C.BASE_CKPT, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    ws: BaseWorkspace = cls(cfg)
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    policy.eval().to(device)
    policy.num_inference_steps = 8
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    wp = mb.load_weights(latest, map_location=device)
    missing, unexpected = policy.load_state_dict(wp["state_dict"], strict=False)
    print(f"[smoke] hot-swap: missing={len(missing)}, unexpected={len(unexpected)}")

    # predict_action 한 번
    with h5py.File(C.BASE_DATASET_PATH, "r") as f:
        d = f["data/demo_0"]
        img = np.array(d["obs/image0"])[:2]
        pose = np.array(d["obs/robot_pose_L"])[:2]
        quat = np.array(d["obs/robot_quat_L"])[:2]
    env_obs = {"image0": img.astype(np.uint8),
               "robot_pose_L": pose.astype(np.float32),
               "robot_quat_L": quat.astype(np.float32)}
    with torch.no_grad():
        od = get_real_obs_dict(env_obs=env_obs, shape_meta=cfg.task.shape_meta)
        od = dict_apply(od, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        out = policy.predict_action(od)
        action = out["action"][0].cpu().numpy()
    print(f"[smoke] hot-swap 후 predict_action OK, action.shape={action.shape}, "
          f"a[0]={action[0,:3].round(4)}")

    print("\n[smoke] ✅ 온라인 루프 전체(에피소드 전송 -> 학습 -> 가중치 발행 -> "
          "hot-swap -> 추론) 검증 완료")
    shutil.rmtree(C.ONLINE_WORKDIR, ignore_errors=True)


if __name__ == "__main__":
    main()
