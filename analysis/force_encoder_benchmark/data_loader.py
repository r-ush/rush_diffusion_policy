"""실제 로봇에서 쌓인 wrench_wrist_R (T,6,32) 윈도우로부터 GRU vs CausalConv
force encoder 벤치마크용 데이터를 만든다.

synthetic 대신 real data를 쓰는 이유:
  online_runs/*/accumulated.hdf5 에는 각 control step마다 250Hz wrench 버퍼의
  최근 32-sample 윈도우가 저장돼 있다. 연속된 두 step의 윈도우는 거의 대부분
  겹치므로(뒤 몇 샘플만 새로 들어옴), 이 겹침을 이용해 원래의 연속 250Hz
  스트림을 거의 그대로 복원할 수 있다. synthetic Gaussian 대신 실제 센서
  노이즈/타이밍을 가진 시계열이 되어 phase-lag 비교가 더 의미있다.

라벨(0/1/2)은 수동 마킹이 없으므로 magnitude/derivative 기반 weak-label로
생성한다 (자유공간 / 지속 접촉(sliding) / 충돌성 스파이크).
"""

from __future__ import annotations

import dataclasses
import glob
from typing import List, Tuple

import h5py
import numpy as np
import zarr


# ---------------------------------------------------------------------------
# stream reconstruction
# ---------------------------------------------------------------------------

def reconstruct_stream(windows: np.ndarray, max_shift: int = 8,
                        err_threshold: float = 0.05) -> np.ndarray:
    """(T, C, W) 윈도우 배열 -> 복원된 연속 스트림 (L, C).

    각 step의 윈도우는 이전 step 윈도우와 몇 샘플만 밀린 채 겹친다. 인접한
    두 윈도우 사이에서 겹치는 정도(shift)를 오차 최소화로 추정하고, 새로
    들어온 부분만 이어붙인다. 겹침을 찾지 못하면(에피소드 경계 등) 윈도우
    전체를 이어붙여 불연속을 인정한다.
    """
    T, C, W = windows.shape
    assert T > 0
    chunks = [windows[0]]  # (C, W)
    for t in range(1, T):
        prev, cur = windows[t - 1], windows[t]
        best_shift, best_err = W, np.inf
        for s in range(1, min(max_shift, W - 1) + 1):
            err = np.abs(prev[:, s:] - cur[:, :W - s]).mean()
            if err < best_err:
                best_err = err
                best_shift = s
        if best_err < err_threshold:
            chunks.append(cur[:, W - best_shift:])
        else:
            chunks.append(cur)  # 겹침을 못 찾음: 불연속 점프로 처리
    stream = np.concatenate(chunks, axis=1)  # (C, L)
    return stream.T  # (L, C)


# ---------------------------------------------------------------------------
# weak labeling
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LabeledStream:
    force: np.ndarray       # (L, 6) Fx,Fy,Fz,Tx,Ty,Tz
    mag: np.ndarray         # (L,) force magnitude (Fx,Fy,Fz norm)
    dmag: np.ndarray        # (L,) |d(mag)/dt| (frame diff)
    labels: np.ndarray      # (L,) int64, 0=free 1=sustained(sliding) 2=collision spike
    source: str             # 어느 hdf5/demo에서 왔는지 (디버그용)


def weak_label(force: np.ndarray, source: str = "", free_pct: float = 30.0,
                spike_pct: float = 95.0, spike_dilate: int = 1) -> LabeledStream:
    """magnitude/derivative percentile threshold 로 프레임별 0/1/2 라벨을 만든다."""
    mag = np.linalg.norm(force[:, :3], axis=1)
    dmag = np.abs(np.diff(mag, prepend=mag[:1]))

    free_thr = np.percentile(mag, free_pct)
    spike_thr = np.percentile(dmag, spike_pct)

    labels = np.zeros(len(mag), dtype=np.int64)
    labels[mag >= free_thr] = 1
    spike_idx = np.where(dmag >= spike_thr)[0]
    for i in spike_idx:
        labels[i:i + 1 + spike_dilate] = 2

    return LabeledStream(force=force, mag=mag, dmag=dmag, labels=labels, source=source)


# ---------------------------------------------------------------------------
# hdf5 loading
# ---------------------------------------------------------------------------

def load_demo_streams(glob_pattern: str, wrench_key: str = "wrench_wrist_R",
                       max_shift: int = 8) -> List[LabeledStream]:
    """glob 패턴에 매칭되는 모든 accumulated.hdf5 의 모든 demo를 복원+라벨링."""
    out = []
    for path in sorted(glob.glob(glob_pattern)):
        try:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    continue
                for demo in f["data"]:
                    key = f"data/{demo}/obs/{wrench_key}"
                    if key not in f:
                        continue
                    windows = f[key][:]  # (T, 6, 32)
                    if windows.shape[0] < 2:
                        continue
                    stream = reconstruct_stream(windows, max_shift=max_shift)
                    src = f"{path}::{demo}"
                    out.append(weak_label(stream, source=src))
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------
# raw zarr episode loading (진짜 연속 250Hz+ 센서 스트림, 겹침 복원 불필요)
# ---------------------------------------------------------------------------
#
# 실제 policy가 보는 wrench는 raw 그대로가 아니다. 실시간 컨트롤러
# (rightarm_hand_with_wrench_encoder_interpolation_controller.py)와 학습 데이터
# 변환 스크립트(data_process/zarr_common_to_diffusion_box_insertion.py) 둘 다
# 동일하게: raw -> 초기 N샘플 평균을 offset으로 빼기 -> EMA(alpha=0.03) 순서로
# 전처리한 뒤 32-length history window로 만든다. 여기서도 같은 전처리를 그대로
# 재현해야 "policy가 실제로 보는 신호"에 대한 분석이 된다.

def subtract_offset(wrench: np.ndarray, offset_samples: int) -> np.ndarray:
    """앞 offset_samples 개의 평균을 offset으로 빼서 gravity/payload bias 제거."""
    if offset_samples <= 0 or len(wrench) == 0:
        return wrench
    n = min(offset_samples, len(wrench))
    offset = np.mean(wrench[:n], axis=0)
    return wrench - offset


def ema_filter(wrench: np.ndarray, alpha: float) -> np.ndarray:
    """지수이동평균 low-pass filter. alpha가 작을수록 smooth (실제 파이프라인 기본 0.03)."""
    if alpha <= 0 or len(wrench) == 0:
        return wrench
    result = np.empty_like(wrench)
    result[0] = wrench[0]
    for idx in range(1, len(wrench)):
        result[idx] = alpha * wrench[idx] + (1.0 - alpha) * result[idx - 1]
    return result


def preprocess_wrench_like_real_pipeline(wrench: np.ndarray, offset_samples: int = 10,
                                          ema_alpha: float = 0.03) -> np.ndarray:
    """zarr_common_to_diffusion_box_insertion.py 의 preprocess_wrench 를 그대로 재현.

    raw -> (초기 offset_samples개로 offset 계산 후 빼기) -> 그 초기 구간 drop
    (실제 컨트롤러도 calibration 동안은 출력을 안 냄) -> EMA(alpha).
    """
    wrench = subtract_offset(wrench, offset_samples)
    if offset_samples > 0:
        drop = min(offset_samples, len(wrench))
        wrench = wrench[drop:]
    wrench = ema_filter(wrench, ema_alpha)
    return wrench


def load_zarr_episode_streams(dataset_dir: str, wrench_name: str = "wrench_raw",
                               limit_episodes: int = 0, apply_filter: bool = True,
                               offset_samples: int = 10, ema_alpha: float = 0.03
                               ) -> List[LabeledStream]:
    """UMIFT-style raw 수집 데이터셋(episode_XXX/ft/<wrench_name>.zarr)을 로드.

    이 zarr는 accumulated.hdf5의 (T,6,32) 윈도우와 달리 겹침 복원 없이 그 자체로
    이미 연속적인 raw FT 센서 스트림(262.5Hz)이라 reconstruct_stream이 필요없다.
    기본값(wrench_name="wrench_raw", apply_filter=True)은 실제 학습/추론 파이프라인과
    동일하게 offset-subtract + EMA(0.03)를 적용한다.
    """
    episode_dirs = sorted(glob.glob(f"{dataset_dir}/episode_*"))
    if limit_episodes:
        episode_dirs = episode_dirs[:limit_episodes]
    out = []
    for ep_dir in episode_dirs:
        zarr_path = f"{ep_dir}/ft/{wrench_name}.zarr"
        try:
            force = zarr.open(zarr_path, mode="r")[:].astype(np.float64)  # (T, 6)
        except (FileNotFoundError, ValueError):
            continue
        if force.shape[0] < 2:
            continue
        if apply_filter:
            force = preprocess_wrench_like_real_pipeline(force, offset_samples, ema_alpha)
            if len(force) < 2:
                continue
        src = f"{ep_dir}::{wrench_name}" + ("+ema0.03" if apply_filter else "+raw")
        out.append(weak_label(force, source=src))
    return out


# ---------------------------------------------------------------------------
# chunking for training
# ---------------------------------------------------------------------------

def make_chunks(stream: LabeledStream, seq_len: int, stride: int
                 ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(seq_len, 6) force / (seq_len,) label 쌍의 리스트로 슬라이딩."""
    L = len(stream.mag)
    chunks = []
    for start in range(0, max(L - seq_len + 1, 0), stride):
        end = start + seq_len
        chunks.append((stream.force[start:end].astype(np.float32),
                        stream.labels[start:end]))
    return chunks
