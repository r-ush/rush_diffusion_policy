"""이미 끝난 eval 결과 폴더에서 infer_obs HDF5를 사후 재구성한다.

왜 필요한가:
  `analysis.modality_attribution.*` 도구는 전부 `--obs episode_XXXXXX_infer_obs.hdf5`
  (= inference마다 policy에 들어간 obs 스냅샷)를 입력으로 받는다. rush의
  online actor / eval 스크립트는 이 덤프를 자동 저장하지만, 다른 repo의 eval
  스크립트로 돌린 결과 폴더(data/results/<run>)에는 덤프가 없다.
  그런 폴더도 replay_buffer.zarr(lowdim + wrench 윈도우)와 videos/{ep}/{cam}.mp4
  (obs 해상도·제어주기, frame k ↔ obs step k 1:1)를 갖고 있으므로 obs를 그대로
  되살릴 수 있다.

두 가지 소스 모드:
  [targets] (기본·정확)  eval_debug/episode_XXXXXX_policy_targets.hdf5 가 있으면
    거기 들어있는 `image/rgb/image0`(추론별 obs 이미지 스택, 무압축)과
    `image/timestamp`를 쓴다. 즉 **정책이 실제로 본 그 프레임**이고, 추론 횟수·시점도
    실제 롤아웃과 1:1로 일치한다. lowdim/wrench는 그 타임스탬프에 가장 가까운
    replay_buffer 스텝에서 가져온다(격자 오차 <0.5 step).
  [video] (폴백)  policy_targets가 없으면 videos/{ep}/{cam}.mp4를 디코드하고
    --stride(=eval의 steps_per_inference) 주기로 obs step을 샘플한다.
    이 경우 H.264 압축 손실 + 추론 시점 근사가 들어간다.

어느 모드든 wrench는 replay_buffer에 저장된 (6,32) 윈도우를 그대로 쓴다
(policy_targets의 6-벡터로는 재현 불가 — README 참고).

사용:
  PY=/home/vision/venv_diffusion/bin/python
  $PY -m analysis.modality_attribution.infer_obs_from_eval_dir \
      -i data/outputs/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt \
      --eval_dir /home/vision/diffusion-policy/data/results/260714_insert_box_hand_rel \
      --stride 6
  # -> <eval_dir>/eval_debug/episode_XXXXXX_infer_obs.hdf5
  # 이후 batch_build_viewers --run_dir <eval_dir> 로 뷰어 생성
"""
import os
import pathlib
import sys

local_paths = [p for p in sys.path if '.local' in p]
sys.path = [p for p in sys.path if '.local' not in p]
import huggingface_hub  # noqa: F401
sys.path = local_paths + sys.path

import click
import dill
import numpy as np
import torch
import zarr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_policy.real_world.real_inference_util import get_real_obs_dict  # noqa: E402
from online_learning.relabel_utils import load_episode_frames  # noqa: E402
from analysis.modality_attribution.record_infer_obs import InferenceObsRecorder  # noqa: E402


def _load_shape_meta(ckpt_path):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu",
                         pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    n_obs_steps = int(cfg.policy.n_obs_steps)
    return cfg.task.shape_meta, n_obs_steps


def _episode_slices(zarr_root):
    episode_ends = np.asarray(zarr_root["meta"]["episode_ends"]).astype(int)
    starts = np.concatenate([[0], episode_ends[:-1]])
    return list(zip(starts.tolist(), episode_ends.tolist()))


def _horizons(shape_meta, n_obs_steps):
    """obs 키별 horizon(=몇 스텝을 policy에 넣는지). 없으면 1."""
    out = dict()
    for key, attr in shape_meta["obs"].items():
        h = attr.get("horizon", None)
        out[key] = int(h) if h is not None else 1
    return out


def _slice_back(arr, i, h):
    """arr에서 [i-h+1 .. i] 구간. 앞이 모자라면 첫 스텝으로 패딩."""
    lo = i - h + 1
    if lo >= 0:
        return arr[lo: i + 1]
    pad = np.repeat(arr[:1], -lo, axis=0)
    return np.concatenate([pad, arr[: i + 1]], axis=0)


def build_episode_infer_obs(zarr_root, eval_dir, ep_index, shape_meta,
                            n_obs_steps, stride, cam_idx, frequency):
    """에피소드 하나의 infer_obs 레코더를 만들어 반환. 데이터 부족 시 None."""
    start, end = _episode_slices(zarr_root)[ep_index]
    T = end - start
    data = zarr_root["data"]
    horizons = _horizons(shape_meta, n_obs_steps)

    lowdim = dict()
    for key in shape_meta["obs"].keys():
        if shape_meta["obs"][key].get("type", "low_dim") == "rgb":
            continue
        if key not in data:
            raise KeyError(
                f"replay_buffer.zarr에 obs 키 '{key}'가 없습니다. "
                f"있는 키: {list(data.keys())}")
        lowdim[key] = np.asarray(data[key][start:end])

    rgb_keys = [k for k, a in shape_meta["obs"].items()
                if a.get("type", "low_dim") == "rgb"]
    timestamps = (np.asarray(data["timestamp"][start:end])
                  if "timestamp" in data else np.arange(T) / float(frequency))

    targets_path = (pathlib.Path(eval_dir) / "eval_debug" /
                    f"episode_{ep_index:06d}_policy_targets.hdf5")
    imgs_by_inference, steps, source = None, None, None

    if targets_path.exists() and len(rgb_keys) == 1:
        import h5py
        with h5py.File(targets_path, "r") as tf:
            rgb_grp = tf.get("image/rgb")
            key0 = rgb_keys[0]
            if rgb_grp is not None and key0 in rgb_grp and "image/timestamp" in tf:
                imgs_by_inference = np.asarray(rgb_grp[key0])       # (N, h, H, W, 3)
                img_ts = np.asarray(tf["image/timestamp"])          # (N, h)
                # 각 추론의 최신 obs 시각 -> 가장 가까운 replay_buffer 스텝
                steps = np.abs(timestamps[None, :] - img_ts[:, -1:]).argmin(axis=1)
                err = np.abs(timestamps[steps] - img_ts[:, -1])
                source = "targets"
                print(f"[infer_obs][ep{ep_index}] policy_targets 사용 "
                      f"(추론 {len(steps)}회, 스텝정렬 오차 max {err.max()*1e3:.0f}ms)")

    if source is None:
        frames = load_episode_frames(
            str(eval_dir), ep_index, n_steps=T,
            cam_idx=cam_idx, frequency=frequency, out_res=None)
        usable = min(T, len(frames))
        max_h = max(list(horizons.values()) + [n_obs_steps])
        if usable < max_h:
            print(f"[infer_obs][ep{ep_index}] 스텝 부족({usable} < {max_h}) → 스킵")
            return None
        steps = np.arange(max_h - 1, usable, stride)
        source = "video"
        print(f"[infer_obs][ep{ep_index}] mp4 폴백 (추론 {len(steps)}회, stride={stride})")

    recorder = InferenceObsRecorder()
    t_start = float(timestamps[0])
    for k, i in enumerate(steps):
        i = int(i)
        env_obs = dict()
        for key in rgb_keys:
            h = horizons[key]
            if source == "targets":
                stack = imgs_by_inference[k]                  # (h, H, W, 3) uint8
                if len(stack) != h:
                    stack = _slice_back(stack, len(stack) - 1, h)
                env_obs[key] = stack
            else:
                env_obs[key] = _slice_back(frames, i, h)
        for key, arr in lowdim.items():
            env_obs[key] = _slice_back(arr, i, horizons[key])

        obs_dict_np = get_real_obs_dict(env_obs=env_obs, shape_meta=shape_meta)
        obs_ts = _slice_back(timestamps, i, n_obs_steps)
        recorder.add(k, obs_dict_np, obs_ts, t_start)

    return recorder


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint (shape_meta 참조용)")
@click.option("--eval_dir", required=True,
              help="replay_buffer.zarr + videos/ 를 가진 eval 출력 폴더")
@click.option("--stride", default=6, type=int, show_default=True,
              help="추론 간격(obs step). eval의 --steps_per_inference와 같게.")
@click.option("--episodes", default="all", show_default=True,
              help="'all' 또는 쉼표구분 인덱스(예: 0,2,4).")
@click.option("--cam_idx", default=0, type=int, show_default=True)
@click.option("--frequency", default=10.0, type=float, show_default=True)
@click.option("--force", is_flag=True, help="이미 infer_obs가 있어도 다시 생성.")
def main(input, eval_dir, stride, episodes, cam_idx, frequency, force):
    eval_dir = pathlib.Path(eval_dir).expanduser()
    zarr_path = eval_dir / "replay_buffer.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"replay_buffer.zarr 없음: {zarr_path}")

    shape_meta, n_obs_steps = _load_shape_meta(input)
    root = zarr.open(str(zarr_path), "r")
    n_ep = len(_episode_slices(root))
    print(f"[infer_obs] {n_ep} episodes, n_obs_steps={n_obs_steps}, stride={stride}")

    if episodes == "all":
        ep_list = list(range(n_ep))
    else:
        ep_list = [int(x) for x in episodes.split(",") if x.strip() != ""]

    debug_dir = eval_dir / "eval_debug"
    written = []
    for ep in ep_list:
        out_path = debug_dir / f"episode_{ep:06d}_infer_obs.hdf5"
        if out_path.exists() and not force:
            print(f"[infer_obs][ep{ep}] 이미 있음 → 스킵 ({out_path.name})")
            continue
        video = eval_dir / "videos" / str(ep) / f"{cam_idx}.mp4"
        if not video.exists():
            print(f"[infer_obs][ep{ep}] 영상 없음 → 스킵 ({video})")
            continue
        rec = build_episode_infer_obs(
            root, eval_dir, ep, shape_meta, n_obs_steps, stride, cam_idx, frequency)
        if rec is None or len(rec) == 0:
            continue
        path = rec.save(str(eval_dir), ep)
        print(f"[infer_obs][ep{ep}] {len(rec)} inferences → {path}")
        written.append(str(path))

    print(f"\n[infer_obs] 완료: {len(written)}개 생성")
    if written:
        print("이제 뷰어 생성:")
        print(f"  python -m analysis.modality_attribution.batch_build_viewers \\\n"
              f"      -i {input} --run_dir {eval_dir}")


if __name__ == "__main__":
    main()
