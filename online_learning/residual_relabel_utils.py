"""
온라인 residual DAgger용 에피소드 relabel (pose-only 6D residual).

full-finetune 경로의 online_learning/relabel_utils.py 대응. 차이:
  * 라벨이 "achieved pose 자체"가 아니라 **slow(base) 예측 대비 잔차(residual)**다.
  * base = actor가 그 스텝에서 실시간으로 돌린 slow policy 예측(slow_pred_target_abs).
    virtual = 사람이 민 결과인 achieved pose(= 한 스텝 뒤 로봇 pose, 프로젝트 관행).
    residual_delta6[t] = delta6_from_base_to_target(slow_pred_target_abs[t], virtual[t]).
  * 손(hand)은 residual 대상이 아니라 obs 입력일 뿐(v1은 pose-only 6D). 손 명령은 base 그대로.

출력 HDF5 구조(FastResidualContextStepDataset + task/hand_online.yaml 이 읽는 포맷):
  data/<demo_name>/obs/{image0, robot_pose_R, robot_quat_R, hand_pose_R, wrench_wrist_R,
                        slow_pred_target_abs, slow_pred_action_rel,
                        residual_delta6_slow_pred_to_virtual}

길이: 마지막 스텝은 "한 스텝 뒤 achieved"가 없어 버린다(T -> T-1). 그래서
hand_online.yaml 은 action_target_shift=0(이 파일이 이미 정렬을 끝냄).
"""
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.residual_policy.pose_util import (
    abs_pose9_to_relative_pose9,
    delta6_from_base_to_target,
    pose6_to_pose9,
    pose_like_to_pose9,
)

# relabel 이 obs 로 그대로 기록하는 원시 관측 키(에피소드 dict 에 반드시 있어야 함)
RAW_OBS_KEYS = ["image0", "robot_pose_R", "robot_quat_R", "hand_pose_R", "wrench_wrist_R"]


def pos_quat_to_pose9(pos, quat):
    """(...,3)+(...,4 xyzw) -> (...,9) pose9 [pos3, rot6d]."""
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    rotvec = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return pose6_to_pose9(np.concatenate([pos, rotvec], axis=-1)).astype(np.float32)


def build_residual_demo(episode: dict):
    """에피소드(per-step numpy 배열 dict) -> residual demo 배열 dict.

    필요한 키:
      RAW_OBS_KEYS (per-step 원시 관측)
      slow_pred_target_abs : (T,9) 또는 (T,10)/(T,7) pose-like. 각 스텝에서 actor 가
                             돌린 slow policy 의 절대 target(base).
      (선택) slow_pred_action_rel : (T,9). 없으면 slow_pred_target_abs 와 현재 pose 로 계산.

    virtual(사람 교정 결과) = 한 스텝 뒤 로봇 achieved pose (robot_pose_R[t+1]).
    """
    for k in RAW_OBS_KEYS + ["slow_pred_target_abs"]:
        if k not in episode:
            raise KeyError(f"episode 에 '{k}' 가 없습니다 (있어야 하는 키: {RAW_OBS_KEYS + ['slow_pred_target_abs']})")

    T = len(episode["robot_pose_R"])
    if T < 2:
        raise ValueError(f"residual relabel 은 최소 2스텝 필요(한 스텝 뒤 achieved 사용), got T={T}")
    n = T - 1  # 마지막 스텝은 achieved(t+1)가 없어 버림

    slow_pred_target_abs = pose_like_to_pose9(np.asarray(episode["slow_pred_target_abs"]))[:n]

    # virtual = 다음 스텝 achieved pose9
    virtual_target_abs = pos_quat_to_pose9(
        episode["robot_pose_R"][1:T],
        episode["robot_quat_R"][1:T],
    )

    residual_delta6 = delta6_from_base_to_target(
        slow_pred_target_abs,
        virtual_target_abs,
    ).astype(np.float32)

    # slow_pred_action_rel: 없으면 현재 pose 기준 상대로 변환
    if "slow_pred_action_rel" in episode:
        slow_pred_action_rel = pose_like_to_pose9(np.asarray(episode["slow_pred_action_rel"]))[:n]
    else:
        current_pose9 = pos_quat_to_pose9(
            episode["robot_pose_R"][:n],
            episode["robot_quat_R"][:n],
        )
        slow_pred_action_rel = abs_pose9_to_relative_pose9(
            current_pose9,
            slow_pred_target_abs,
        ).astype(np.float32)

    demo = {}
    for k in RAW_OBS_KEYS:
        demo[k] = np.asarray(episode[k])[:n]
    demo["slow_pred_target_abs"] = slow_pred_target_abs.astype(np.float32)
    demo["slow_pred_action_rel"] = slow_pred_action_rel.astype(np.float32)
    demo["virtual_target_abs"] = virtual_target_abs.astype(np.float32)
    demo["residual_delta6_slow_pred_to_virtual"] = residual_delta6
    return demo


def write_residual_episode_hdf5(out_path, episode: dict, demo_name="demo_0"):
    """단일 에피소드를 residual 포맷 HDF5(data/<demo_name>/obs/...)로 저장."""
    demo = build_residual_demo(episode)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        obs_grp = f.create_group("data").create_group(demo_name).create_group("obs")
        for k, v in demo.items():
            obs_grp.create_dataset(k, data=np.asarray(v))
        f["data"].attrs["num_demos"] = 1
    return out_path
