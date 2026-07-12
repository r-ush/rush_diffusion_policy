"""Drop-in recorder: rollout 중 inference마다 들어간 obs를 그대로 저장.

왜 필요한가:
  policy는 wrench를 (C, 32) history window로 소비한다(bae_real_env_...py:412).
  하지만 기존 debug HDF5(episode_XXXXXX_policy_targets.hdf5)에는 이 윈도우가 아니라
  action 시점에 샘플된 6-벡터만 들어있다. 따라서 그 파일만으로는 wrench conditioning을
  충실히 재현할 수 없다. 이 recorder는 predict_action에 실제로 들어간 obs_dict_np를
  통째로 저장해서, replay_offline.py가 정확히 같은 입력으로 다시 돌릴 수 있게 한다.

eval 스크립트(bae_eval_real_robot_rightarm_insert_plug.py) 통합 방법 (3곳):

  # (a) episode 시작부, image_debug_records 만드는 근처
  from analysis.modality_attribution.record_infer_obs import InferenceObsRecorder
  obs_recorder = InferenceObsRecorder()

  # (b) inference loop 안, obs_dict_np 완성 직후(add_wrench_obs_noise 다음, dict_apply 전)
  obs_recorder.add(inference_index, obs_dict_np, obs_timestamps, eval_t_start)

  # (c) 에피소드 종료 저장부(_finish_episode_and_save_diagnostics 근처)
  obs_recorder.save(output, episode_id)

저장 위치: <output>/eval_debug/episode_XXXXXX_infer_obs.hdf5
"""

from __future__ import annotations

import pathlib
from typing import Dict, List

import numpy as np


DEBUG_DIR_NAME = "eval_debug"


class InferenceObsRecorder:
    """inference별 obs_dict_np 스냅샷을 모아 HDF5로 저장한다."""

    def __init__(self):
        self._records: List[dict] = []

    def add(self, inference_index: int, obs_dict_np: Dict[str, np.ndarray],
            obs_timestamps=None, eval_t_start=None):
        snap = {k: np.asarray(v).copy() for k, v in obs_dict_np.items()}
        self._records.append({
            "inference_index": int(inference_index),
            "obs": snap,
            "obs_timestamps": (
                np.asarray(obs_timestamps, dtype=np.float64)
                if obs_timestamps is not None else np.empty((0,), dtype=np.float64)
            ),
            "elapsed_s": (
                float(np.asarray(obs_timestamps, dtype=np.float64)[-1] - eval_t_start)
                if (obs_timestamps is not None and eval_t_start is not None
                    and len(np.asarray(obs_timestamps)) > 0)
                else float("nan")
            ),
        })

    def __len__(self):
        return len(self._records)

    def save(self, output_dir, episode_id) -> "pathlib.Path | None":
        if episode_id is None or len(self._records) == 0:
            return None
        import h5py

        debug_dir = pathlib.Path(output_dir).joinpath(DEBUG_DIR_NAME)
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir.joinpath(f"episode_{episode_id:06d}_infer_obs.hdf5")

        # obs 키는 모든 inference에 공통이라고 가정(같은 policy). 없는 키는 스킵.
        common_keys = set(self._records[0]["obs"].keys())
        for rec in self._records[1:]:
            common_keys &= set(rec["obs"].keys())
        common_keys = sorted(common_keys)

        with h5py.File(path, "w") as f:
            f.attrs["schema"] = (
                "Per-inference obs_dict_np snapshots fed to policy.predict_action "
                "(post get_real_obs_dict + wrench noise, pre batch-dim). "
                "obs/<key> layout: N_inference, <native obs shape>."
            )
            f.attrs["n_inference"] = len(self._records)
            f.create_dataset(
                "inference_index",
                data=np.asarray([r["inference_index"] for r in self._records], dtype=np.int64),
            )
            f.create_dataset(
                "elapsed_s",
                data=np.asarray([r["elapsed_s"] for r in self._records], dtype=np.float64),
            )
            # obs_timestamps는 길이가 일정하면 저장
            ts_lens = {len(r["obs_timestamps"]) for r in self._records}
            if len(ts_lens) == 1 and ts_lens != {0}:
                f.create_dataset(
                    "obs_timestamps",
                    data=np.stack([r["obs_timestamps"] for r in self._records], axis=0),
                )

            grp = f.create_group("obs")
            grp.attrs["keys"] = np.asarray(common_keys, dtype="S")
            for key in common_keys:
                stacked = np.stack([r["obs"][key] for r in self._records], axis=0)
                # 이미지 등 큰 배열은 압축
                compress = stacked.nbytes > (1 << 20)
                grp.create_dataset(
                    key,
                    data=stacked,
                    compression="gzip" if compress else None,
                    compression_opts=4 if compress else None,
                )

        print(f"Inference obs snapshots saved: {path}")
        return path


def load_inference_obs(path):
    """저장된 infer_obs HDF5를 읽어 inference 리스트로 반환.

    반환: dict(inference_index, elapsed_s, obs_keys, obs_by_inference[list of dict])
    """
    import h5py

    with h5py.File(path, "r") as f:
        inference_index = np.asarray(f["inference_index"])
        elapsed_s = np.asarray(f["elapsed_s"])
        grp = f["obs"]
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in grp.attrs["keys"]]
        obs_stacks = {key: np.asarray(grp[key]) for key in keys}

    n = len(inference_index)
    obs_by_inference = []
    for i in range(n):
        obs_by_inference.append({key: obs_stacks[key][i] for key in keys})

    return {
        "inference_index": inference_index,
        "elapsed_s": elapsed_s,
        "obs_keys": keys,
        "obs_by_inference": obs_by_inference,
    }
