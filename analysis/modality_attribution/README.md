# Modality Attribution

롤아웃 중 policy의 action이 **어느 modality(vision / wrench(force) / low_dim)에 더 좌우되는지**
진단하기 위한 도구 모음. `bae_eval_real_robot_rightarm_insert_plug.py` 계열의
`*_wrench_encoder_*` policy(vision encoder + force encoder를 global_cond로 fusion)를 대상으로 한다.

## 왜 이런 구조인가 (중요)

- 이 policy들은 vision feature + low_dim + force(wrench) feature를 하나의 `global_cond`
  벡터로 만들어 `ConditionalUnet1D`에 FiLM conditioning으로 넣는다
  (`predict_action` → `nobs_features` → `global_cond`).
  그래서 **어느 modality를 얼마나 지웠을 때 action이 얼마나 바뀌는지**를 재면 기여도를 분리할 수 있다.
- policy는 wrench를 `(C, 32)` **history window**로 소비한다(`bae_real_env_...py:412`).
  기존 debug HDF5(`episode_XXXXXX_policy_targets.hdf5`)에는 이 윈도우가 아니라 action 시점에
  샘플된 6-벡터만 저장돼 있어서 **그 파일만으로는 충실한 재현이 불가능**하다.
  → 그래서 rollout 때 inference obs를 통째로 덤프하는 recorder를 먼저 붙여야 한다.

## 디렉토리 구성

| 파일 | 역할 |
|---|---|
| `attribution.py` | 핵심 라이브러리: seed 고정 예측, `action_delta`, baseline 빌더(`make_zero_wrench`/`make_blank_vision`/`make_replace_low_dim`/`make_freeze_vision`), `ablation_deltas`, gradient/attention |
| `record_infer_obs.py` | rollout 때 inference obs를 그대로 저장하는 recorder (**online actor엔 내장·자동 저장**, rush의 eval 스크립트도 호출) |
| `infer_obs_from_eval_dir.py` | recorder 없이 받은 eval 결과 폴더에서 infer_obs를 **사후 재구성**(replay_buffer.zarr의 lowdim·wrench 윈도우 + policy_targets.hdf5의 실제 obs 이미지) |
| `replay_offline.py` | 오프라인 modality Δ 시간축 그래프/CSV (vision vs wrench) |
| `visualize_attribution.py` | 정적 PNG: **vision occlusion saliency 몽타주** + **force 축별(Fx~Tz) 히트맵** |
| `build_attribution_viewer.py` | **★ 인터랙티브 HTML 뷰어**: 프레임 슬라이더 + saliency + force축 막대 + **3-way dominance(vision/wrench/joint)** |
| `batch_build_viewers.py` | eval_debug 스캔 → 뷰어 없는 에피소드만 자동 생성 (`--loop`로 감시 모드) |
| `combine_pngs.py` | 여러 PNG를 세로/가로로 결합 |
| `analyze_force_influence.py` | force 영향이 작은지 진단(민감도 스윕 + 공정 baseline 비교) |
| `analyze_wrench_bottleneck.py` | wrench 저조 **원인 메커니즘 분해**(토큰 변동성 / attention / projection 가중치) |
| `README.md` | 이 문서 |

## ⚡ 빠른 사용법 (커맨드 요약)

```bash
PY=/home/vision/venv_diffusion/bin/python                                        # timm+imagecodecs 있는 env
CKPT=data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt
RUN=data/online_runs/run_hand/actor_episodes

# 0) 데이터: online actor를 --use_hand로 돌리고 에피소드를 '유지 종료'하면
#    $RUN/eval_debug/episode_XXXXXX_infer_obs.hdf5 가 자동 저장됨 (recorder 내장).
#
# 0-b) recorder가 없는 eval로 이미 받아둔 결과 폴더(data/results/<run>)를 분석하려면
#      infer_obs를 사후 재구성한다 (replay_buffer.zarr + policy_targets.hdf5 이용).
$PY -m analysis.modality_attribution.infer_obs_from_eval_dir \
    -i $CKPT --eval_dir /path/to/data/results/<run>
#      -> <eval_dir>/eval_debug/episode_XXXXXX_infer_obs.hdf5 생성. 이후 아래 도구 전부 동일하게 사용.
#      (검증: 재구성 obs로 재예측한 action이 롤아웃이 기록한 virtual target과 평균 ~2mm 일치)

# 1) ★ 새 에피소드 분석자료(뷰어) 자동 생성 — 뷰어 없는 것만 만들고 있는 건 스킵
$PY -m analysis.modality_attribution.batch_build_viewers -i $CKPT
#    감시 모드(롤아웃 끝날 때마다 자동 생성): 끝에 --loop 30 추가

# 2) 뷰어 열기 (브라우저, 데이터·이미지가 파일 하나에 embed — 외부 전송 없음)
xdg-open $RUN/attribution_ep000017/viewer.html

# ── 단일 에피소드 / 저수준 도구 ──
# 인터랙티브 뷰어 하나만
$PY -m analysis.modality_attribution.build_attribution_viewer \
    -i $CKPT --obs $RUN/eval_debug/episode_000017_infer_obs.hdf5 \
    -o $RUN/attribution_ep000017/viewer.html --grid 8 --seeds 0,1 --frames all
# 정적 PNG(saliency 몽타주 + force 축 히트맵)
$PY -m analysis.modality_attribution.visualize_attribution -i $CKPT --obs <infer_obs> -o <dir>/detail
# 시간축 Δ 그래프/CSV
$PY -m analysis.modality_attribution.replay_offline -i $CKPT --obs <infer_obs> -o <dir>/attribution
# force 영향 진단 / wrench 병목 메커니즘
$PY -m analysis.modality_attribution.analyze_force_influence  -i $CKPT --obs <infer_obs>
$PY -m analysis.modality_attribution.analyze_wrench_bottleneck -i $CKPT --obs <infer_obs>
```

### 뷰어 읽는 법
- **Modality dominance 배지**: 그 프레임 최강 modality — 🟢JOINT(proprioception) / 🔵VISION / 🔴WRENCH.
- **Δvision/Δwrench/Δjoint**: 해당 modality를 공정 baseline으로 지웠을 때 action 변화(클수록 의존).
- **좌측 saliency**: 이미지의 어디를 보나(빨강=가리면 action 크게 바뀜). **force 막대**: 어느 축(Fx~Tz)이 중요.
- **하단 타임라인**: 3색 Δ 곡선(초록/파랑/빨강), 클릭=그 프레임 점프.

## 방법 4가지 (필요하면 바꿔가며 쓰기)

| # | 방법 | 함수 | 장점 | 한계 | fuse_mode |
|---|---|---|---|---|---|
| 1 | **Counterfactual ablation** (기본) | `ablation_deltas` | 해석이 가장 명확, mode 무관 | baseline 선택에 민감 | 전부 |
| 2 | Gradient saliency | `gradient_saliency` | inference 1회로 구간별 민감도, 실시간 로깅 가능 | 국소적 근사 | concat only |
| 3 | Wrench noise 주입 | (eval의 `--wrench_noise_*` 옵션) | 이미 존재하는 인프라 | per-step 분해 불가, 에피소드 단위 | 전부 |
| 4 | Attention weight | `capture_modality_attention` | 토큰별 attention 관찰 | attention≠attribution, 교차검증용 | modality-attention only |

핵심 원칙 두 가지:
1. **diffusion sampling seed를 고정**한다. 안 그러면 Δ가 "modality 차이"가 아니라 "noise 차이"를 잰다.
   (`predict_action`이 `**self.kwargs`로 generator를 받으므로 여기에 주입 → 결정적)
2. action 비교는 **normalized 공간(total)** 에서 하되, 위치는 **물리 단위(pos, m)** 로도 함께 잰다.

## 사용법

### 1단계 — rollout 때 obs 덤프 (한 번만 eval 스크립트에 3줄 추가)

`bae_eval_real_robot_rightarm_insert_plug.py`에 아래 3곳을 넣는다
(자세한 위치는 `record_infer_obs.py` 상단 docstring 참고):

```python
# (a) 에피소드 시작부
from analysis.modality_attribution.record_infer_obs import InferenceObsRecorder
obs_recorder = InferenceObsRecorder()

# (b) inference loop 안, add_wrench_obs_noise 직후 / dict_apply(텐서화) 직전
obs_recorder.add(inference_index, obs_dict_np, obs_timestamps, eval_t_start)

# (c) 에피소드 종료 저장부(_finish_episode_and_save_diagnostics 근처)
obs_recorder.save(output, episode_id)
```

저장 결과: `<output>/eval_debug/episode_XXXXXX_infer_obs.hdf5`

> 이 recorder는 로봇/제어에 전혀 개입하지 않고 obs 스냅샷만 모은다. 기존 rollout 동작은 그대로.

### 2단계 — 오프라인 분석 (로봇 없이)

```bash
python -m analysis.modality_attribution.replay_offline \
    -i data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt \
    --obs data/results/260710_insert_box_hand/eval_debug/episode_000000_infer_obs.hdf5 \
    -o   data/results/260710_insert_box_hand/attribution
```

주요 옵션:
- `--seeds 0,1,2` : 평균낼 sampling seed (많을수록 안정적, 느려짐)
- `--vision_baseline start|self` : `start`=에피소드 첫 프레임 고정("화면이 안 바뀌었다면"), `self`=자기 마지막 프레임 고정
- `--wrench_baseline zero` : wrench를 0(무접촉)으로("접촉이 없었다면")
- `--gradient` : gradient saliency(concat mode)도 함께 계산
- `--num_inference_steps 16` : DDIM step 수(rollout과 동일하게)
- `--limit N` : 앞 N개 inference만

출력:
- `attribution.csv` — inference별 Δvision/Δwrench (total / pos(m) / rot), 그리고 dominance share
- `attribution_timeline.png` — 시간축 3-panel: normalized Δ / 위치 Δ(m) / vision dominance share
- `gradient_saliency.csv` — `--gradient` 시

읽는 법: **접근(approach) 구간에서는 vision Δ가 크고, 접촉 후 삽입 구간에서 wrench Δ가 커지는
phase 전환**이 보이면 "삽입은 force에 의존한다"는 해석이 뒷받침된다.

## 라이브러리로 직접 쓰기 (다른 방법으로 갈아탈 때)

```python
from analysis.modality_attribution import attribution as attr
from analysis.modality_attribution.replay_offline import load_policy

policy, cfg = load_policy(ckpt_path, num_inference_steps=16)
obs_dict = attr.obs_np_to_tensor(obs_np, policy.device)   # obs_np: recorder 스냅샷 1개

# 1) ablation
res = attr.ablation_deltas(policy, obs_dict, {
    "wrench": attr.make_zero_wrench(policy),
    "vision": attr.make_freeze_vision(policy, obs_dict),
}, seeds=(0,1,2))
print({k: v.as_dict() for k, v in res.deltas.items()})

# 2) gradient saliency (concat mode)
print(attr.gradient_saliency(policy, obs_dict))

# 4) attention (modality-attention mode)
print(attr.capture_modality_attention(policy, obs_dict))
```

## baseline 확장 아이디어 (더 semantic한 반사실)

- wrench → 에피소드 초반 free-space 구간의 실제 wrench로 교체(현재는 0). `make_replace_low_dim` 패턴 참고.
- vision → 데이터셋 평균 이미지 / blur (`make_replace_vision` 사용).
- feature-level ablation → `compute_feature_layout`로 `global_cond`의 force 구간만 학습 데이터 평균
  feature로 치환하면 encoder 재통과 없이 빠르게 잴 수 있다(concat mode).

## 한계 / 주의

- **덤프가 없으면 못 돈다.** 기존 `policy_targets.hdf5`에는 wrench history window가 없어 재현 불가.
- ablation baseline이 학습 분포 밖(OOD)이면 Δ가 과장될 수 있다. seed 여러 개 + 여러 baseline 교차확인 권장.
- gradient/attention은 근사다. 결론은 ablation(#1)을 1차로, 나머지는 교차검증으로.

## 결과 해석 & 발견 (중요, 2026-07)

### 공정 baseline (freeze-to-start 금지)
- **vision baseline은 `mean`(중립 이미지로 교체)이 공정**이다 = zero-wrench의 대응. 도구 기본값도 `mean`.
- `freeze-to-start`(첫 프레임 고정)는 로봇이 움직일수록 Δ가 끝없이 커져 **vision을 과대평가**한다(초기 오해의 원인).
- `self`(직전 프레임 고정)는 2프레임 광류만 재므로 **과소평가** — vision 의존도의 척도가 아님.
- 축별 공정성 요약: vision=`make_blank_vision` / wrench=`make_zero_wrench` / joint=평균 low_dim(`make_replace_low_dim`).

### 3-way dominance (joint 추가)
- vision·wrench만 보면 애매 → **joint(=`robot_pose_R`+`robot_quat_R`+`hand_pose_R`)** 추가해 3-way로 본다.
- 이 정책(abs action)은 대체로 **joint ≳ vision ≫ wrench**. abs 목표를 내려면 현재 pose가 필수라 joint가 최대.

### wrench 영향이 낮은 원인 (`analyze_wrench_bottleneck.py`로 실측)
- 이 ckpt는 **`modality-attention` fusion**: 토큰 `[vis_t0, vis_t1, wrench]` 각 512차원(동일).
- **concat 크기 문제 아님**(동일 512차원), **projection 가중치도 wrench가 오히려 큼**(공정).
- **진짜 병목**: wrench 토큰 출력 변동성이 vision의 ~1/7 (std 0.0018 vs 0.012) → **유효 기여 ≈ 8.6%**.
  = 인코더가 force를 사실상 상수로 뭉갬(**virtual-target action이 pose+vision으로 결정돼 force→action 신호가 약함**).
- **개선 방향**(우선순위): ①학습신호(vision **modality dropout** + force-reactive correction 데이터 + auxiliary force 예측) → ②토큰 **LayerNorm/gain**으로 스케일 균형 → ③명시적 `action=base+Δ(force)` 경로 + 접촉 가중 loss → ④dilated causal conv.
- 개선 후 **같은 attribution 툴로 Δwrench 재측정**해 검증하는 루프를 권장.
