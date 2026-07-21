#!/usr/bin/env python
"""
옛 full-finetune 교정 데이터를 residual online learner 에 **하나씩 흘려보내는** actor 대역.

목적: 로봇 없이, 예전에 쌓아둔 correction 에피소드(raw obs + achieved)를 새 cr-dagger
residual 방식의 **온라인 루프(warm-continue)**로 재현한다. (사용자 요청 B)

동작(에피소드마다):
  1. 옛 에피소드의 raw obs 를 읽음(image0, robot_pose_R, robot_quat_R, hand_pose_R, wrench).
  2. **slow(base) policy 를 그 obs 에 돌려 slow_pred_target_abs 를 프레임별로 재계산**
     (옛 데이터엔 slow_pred 가 없으므로; create_slow_pred_fast_dataset 로직 재사용).
  3. residual 포맷 HDF5 로 만들어(mailbox 가 기대하는 형태) send_episode.
  4. delay 후 다음 에피소드 — 그 사이 learner 가 warm-continue 로 head 를 학습·발행.

즉 residual_teleop_learner.py 는 그대로 돌리고, 이 스크립트가 actor 를 대신해 실데이터를
스트리밍한다. → "잘못된 상황에서 residual 이 교정 쪽으로 당기는지"를 verify 스크립트로 검증.

실행(별도 터미널에서 learner 를 먼저 띄운 상태에서):
  RESIDUAL_ONLINE_WORKDIR=data/online_runs/run_hand_residual_abs \
  <py> online_learning/replay_feed_episodes.py \
     --slow_ckpt <260710 abs ckpt> \
     --source /media/.../online_runs/run_hand/transitions \
     --delay 20 --max_episodes 0
"""
import os
import sys
import glob
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
import torch

from diffusion_policy.residual_policy.create_slow_pred_fast_dataset import (
    load_policy_from_ckpt, build_policy_obs, collate_policy_obs,
    current_pose9_for_step, slow_action_to_abs,
)
from online_learning import config_residual_online as C
from online_learning.mailbox import FileMailbox
from online_learning.residual_relabel_utils import write_residual_episode_hdf5, RAW_OBS_KEYS


def _sorted_demo_keys(g):
    return sorted(g.keys(), key=lambda s: int(s.split("_")[-1]))


def iter_source_episodes(source):
    """source 가 디렉토리면 *.hdf5 전체, 파일이면 그 파일. 각 data/demo_* 를 (name, obs_dict) 로 yield."""
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "*.hdf5")))
    else:
        files = [source]
    for fp in files:
        with h5py.File(fp, "r") as f:
            if "data" not in f:
                continue
            for dm in _sorted_demo_keys(f["data"]):
                d = f["data"][dm]
                if "obs" not in d:
                    continue
                obs = {k: np.asarray(d["obs"][k]) for k in d["obs"].keys()}
                tag = f"{os.path.basename(fp)}:{dm}"
                yield tag, obs


@torch.no_grad()
def compute_slow_pred_target_abs(slow_policy, obs_dict, device, arm="R",
                                 action_pose_repr="abs", batch_size=16):
    """각 프레임 t 에서 slow policy 를 돌려 다음 목표(첫 스텝)를 abs pose9 로.
    obs_dict: {rgb/low_dim/wrench key -> (T, ...)} numpy. build_policy_obs 가 dict 도 인덱싱 가능."""
    T = len(obs_dict[f"robot_pose_{arm}"])
    out = np.zeros((T, 9), dtype=np.float32)
    for start in range(0, T, batch_size):
        idxs = list(range(start, min(T, start + batch_size)))
        slow_obs = collate_policy_obs([build_policy_obs(slow_policy, obs_dict, t, device) for t in idxs])
        slow_action = slow_policy.predict_action(slow_obs)["action"][:, 0].detach().cpu().numpy()  # (B,16)
        for j, t in enumerate(idxs):
            cur = current_pose9_for_step(obs_dict, t, arm)
            out[t] = slow_action_to_abs(slow_action[j][:9], cur, action_pose_repr)
    return out


def main():
    ap = argparse.ArgumentParser(description="옛 교정 데이터를 residual learner 에 하나씩 흘려보내는 actor 대역.")
    ap.add_argument("--slow_ckpt", default=None, help="slow(base) ckpt. 미지정 시 C.SLOW_CKPT.")
    ap.add_argument("--source", required=True, help="옛 에피소드 디렉토리(*.hdf5) 또는 accumulated.hdf5.")
    ap.add_argument("--delay", type=float, default=20.0, help="에피소드 간 대기(초). learner 가 학습할 시간.")
    ap.add_argument("--max_episodes", type=int, default=0, help="0=전체. 테스트 시 몇 개만.")
    ap.add_argument("--num_inference_steps", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--arm", default="R")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    slow_ckpt = args.slow_ckpt or C.SLOW_CKPT
    device = torch.device(args.device or (C.DEVICE if torch.cuda.is_available() else "cpu"))
    mailbox = FileMailbox(C.ONLINE_WORKDIR)

    print(f"[Feeder] slow ckpt 로드: {slow_ckpt}")
    _, slow_policy = load_policy_from_ckpt(slow_ckpt, device=device, use_ema=True)
    slow_policy.num_inference_steps = args.num_inference_steps
    slow_policy.n_action_steps = int(slow_policy.horizon) - int(slow_policy.n_obs_steps) + 1
    action_pose_repr = getattr(slow_policy, "action_pose_repr", "abs")
    print(f"[Feeder] action_pose_repr={action_pose_repr}, n_obs={slow_policy.n_obs_steps}, "
          f"n_action={slow_policy.n_action_steps}, workdir={C.ONLINE_WORKDIR}")

    sent = 0
    for tag, obs in iter_source_episodes(args.source):
        # residual relabel 이 요구하는 raw obs 키가 다 있나 확인
        missing = [k for k in RAW_OBS_KEYS if k not in obs]
        if missing:
            print(f"[Feeder] {tag} 스킵(키 없음: {missing})")
            continue
        T = len(obs[f"robot_pose_{args.arm}"])
        if T < 2:
            print(f"[Feeder] {tag} 스킵(T={T})")
            continue

        t0 = time.time()
        slow_pred = compute_slow_pred_target_abs(
            slow_policy, obs, device, arm=args.arm,
            action_pose_repr=action_pose_repr, batch_size=args.batch_size)

        episode = {k: obs[k] for k in RAW_OBS_KEYS}
        episode["slow_pred_target_abs"] = slow_pred
        out = write_residual_episode_hdf5(
            os.path.join(C.ONLINE_WORKDIR, "feeder_last.hdf5"), episode, demo_name="demo_0")
        mailbox.send_episode(out)
        sent += 1
        print(f"[Feeder] 전송 #{sent}: {tag}  (T={T}, slow_pred {time.time()-t0:.1f}s)")

        if args.max_episodes and sent >= args.max_episodes:
            break
        if args.delay > 0:
            time.sleep(args.delay)

    print(f"[Feeder] 완료: {sent}개 에피소드 전송.")


if __name__ == "__main__":
    main()
