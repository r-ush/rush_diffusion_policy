# GRU vs Causal Conv force encoder benchmark (실측 데이터 기반)

`analysis/modality_attribution/`에서 관측된 "wrench encoder(CausalConv)가 힘 신호를
거의 상수로 collapse한다"는 문제와 별개로, **인코더 아키텍처 자체의 특성**
(반응 지연, 처리 속도)을 synthetic이 아닌 **실제 로봇 wrench 데이터**로 비교한다.

## 데이터 소스 (두 가지)

1. **`--data-glob` (기본, hdf5)**: `online_runs/*/accumulated.hdf5`의
   `obs/wrench_wrist_R`는 매 control step마다 250Hz wrench 버퍼의 최근
   32-sample 윈도우 `(T, 6, 32)`를 저장한다. 인접한 두 step의 윈도우는 거의 다
   겹치므로(뒤 몇 샘플만 새로 들어옴), 이 겹침을 이용해 원래의 연속 스트림을
   복원한다 (`data_loader.reconstruct_stream`). demo 수가 적다(3~18개).
2. **`--zarr-dir` (권장, UMIFT raw 수집)**: 예) `/home/rush/Desktop/Datasets/20260630_195919`.
   `episode_XXX/ft/wrench_raw.zarr`가 그 자체로 이미 연속적인 raw FT 센서
   스트림(262.5Hz)이라 겹침 복원이 필요 없다. 100개 에피소드, 에피소드당
   ~5000~6000 샘플(~20~24초)로 데이터가 훨씬 풍부하다.

라벨(0=자유공간 / 1=지속 접촉(sliding) / 2=충돌성 스파이크)은 수동 마킹이 없어서
magnitude/derivative percentile 기반 weak-label로 생성한다 (`data_loader.weak_label`).

## 실제 policy가 보는 신호로 맞추기 (offset-subtract + EMA)

`--zarr-dir` 경로는 기본적으로 **실제 학습/추론 파이프라인과 동일한 전처리**를
적용한다 (`data_loader.preprocess_wrench_like_real_pipeline`은
`data_process/zarr_common_to_diffusion_box_insertion.py`의 `preprocess_wrench`를
그대로 재현):

```
raw (wrench_raw.zarr) → 초기 offset_samples(기본 10)개 평균을 offset으로 빼기
                       → 그 초기 구간 drop
                       → EMA(alpha=0.03) low-pass filter
```

실시간 컨트롤러(`diffusion_policy/real_world/rightarm_hand_with_wrench_encoder_interpolation_controller.py`,
`WRENCH_EMA_ALPHA = 0.03`, 250Hz)도 정확히 같은 순서로 처리하므로, 이 전처리를
빼면 policy가 실제로 한 번도 보지 못하는 raw 노이즈를 분석하게 된다.
`--no-filter`로 끄면 raw 그대로(비교용)도 가능하고, `--wrench-ema-alpha`/
`--wrench-offset-samples`로 값도 바꿀 수 있다.

## 파일

- `data_loader.py` — hdf5/zarr 로딩, 스트림 복원(`reconstruct_stream`), 실파이프라인
  전처리(`preprocess_wrench_like_real_pipeline`), weak labeling, 학습용 chunk 생성
- `models.py` — `GRUEncoder` (2-layer GRU + MLP head), `CausalConvEncoder`
  (dilated causal 1D-CNN 3층, 왼쪽만 padding하는 `CausalConv1d`)
- `benchmark.py` — 학습 → phase-lag 평가 → forward-pass latency 스윕 → PNG 리포트

## 사용

```
# UMIFT raw 데이터셋 (권장, EMA 전처리 기본 적용)
python -m analysis.force_encoder_benchmark.benchmark \
    --zarr-dir /home/rush/Desktop/Datasets/20260630_195919 \
    --eval-holdout 5 --epochs 30 --seq-len 128 --stride 32

# 이전 online_runs hdf5 방식
python -m analysis.force_encoder_benchmark.benchmark \
    --data-glob "/media/rush/.../online_runs/run_hand*/accumulated.hdf5" \
    --epochs 30 --seq-len 128 --stride 16
```

출력 (`analysis/force_encoder_benchmark/outputs/`):
- `gru_vs_causalconv_report.png` — 충돌 onset 근방 P(collision) 곡선(axis별 Fx/Fy/Fz
  포함) + latency 곡선 + 감지 지연 히스토그램 + 텍스트 요약
- `force_axis_breakdown.png` — 평가 에피소드 전체 구간의 Fx/Fy/Fz + free/sliding/
  collision 라벨 배경 (어느 축이 어떤 상태를 특징짓는지)

## 관측된 결과

### v1 — online_runs hdf5, 18 demo, raw (필터 없음)
- **Latency**: CausalConv 0.09→0.17ms, GRU 0.09→0.89ms (seq_len=1024에서 ~5배)
- **Phase lag**: GRU 평균 2.4 samples(~9.6ms) vs CausalConv 10.1 samples(~40.4ms) —
  통념(GRU가 느리다)과 반대. 데이터가 적어(3 demo eval) 신뢰도는 낮음.

### v2 — UMIFT zarr, 100 episode, `jt_tared_wrench`(raw, 필터 없음)
- **Latency**: 동일 패턴 재확인.
- **Phase lag**: 둘 다 평균 ~0.25 samples(~1ms), 거의 즉각 반응, missed 2/972 —
  데이터가 많고 반복적인 태스크라 두 모델 다 사실상 완벽하게 학습됨. 아키텍처
  차이보다 데이터 양이 lag를 더 크게 좌우한다는 것을 보여줌.
- **axis breakdown**: 에피소드 시작 후 짧은 baseline 이후로는 거의 전체가
  sliding/collision — "가끔 충돌하는 자유공간 이동"이 아니라 "거의 항상 접촉
  중인 연속 조작" 태스크라는 게 드러남.
- 단, 이 결과는 **raw(unfiltered)** 신호 기준이라 실제 policy가 보는 신호와
  다르다 (아래 v3 참고).

### v3 — UMIFT zarr, 100 episode, `wrench_raw` + offset-subtract + EMA(0.03) (실제 파이프라인과 동일)
- **Latency**: 필터와 무관 (아키텍처 고유 특성), 동일 패턴.
- **event 수**: EMA가 급격한 변화를 스무딩해서 972 → **137개**로 급감 — policy가
  실제로 보는 신호에는 저렇게 날카로운 스파이크가 애초에 없다는 뜻.
- **Phase lag**: GRU 평균 1.18 samples(~4.7ms) missed 5/137, CausalConv 평균
  0.44 samples(~1.7ms) missed 13/137 — CausalConv가 잡아내는 것에 대해서는
  더 빠르지만 놓치는 이벤트가 더 많음(빠르지만 덜 안정적 vs 약간 느리지만
  더 안정적인 trade-off).
- **axis breakdown**: 신호가 매끈해져 실제 삽입 동작의 물리적 형태(긴 자유구간 →
  완만한 접촉 진입 → Fz 골짜기 → 안정적 sliding plateau)가 명확히 드러남.

결론적으로 **필터 유무가 결과를 완전히 바꾼다** — 아키텍처 비교를 할 때는 항상
policy가 실제로 보는 전처리된 신호로 맞춰야 한다 (기본값이 이미 그렇게 설정됨).
