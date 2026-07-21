# Residual-Online DAgger — 로봇 실행 런북 (다음에 완성/실행할 때 보는 문서)

> 이 문서는 **로봇 앞에서** residual-online DAgger를 완성·실행할 때의 체크리스트다.
> 설계/배경은 `RESIDUAL_ONLINE_HANDOFF.md`, 학습 코어(검증됨)는 `residual_teleop_learner.py`.
> 현재 상태: **learner 코어 = 완성·검증됨(로봇 없이 통과). actor = 초안 작성됨
> (`residual_teleop_actor_env_runner.py`), 단 하드웨어(rclpy/realsense/servo) 없이는 실행
> 검증 불가 → 로봇에서 [VERIFY] 지점만 실측 확인·튜닝하면 됨.**
> 즉 로봇에서 할 일은 (1) actor [VERIFY] 확인 → (2) 멀티터미널 실행 → (3) 교정 루프 → (4) 검증.

---

## 0. 사전 체크 (로봇 세션 시작 시 제일 먼저)

```bash
# (a) env — 반드시 timm 있는 bae_robodiff
PY=/home/rush/anaconda3/envs/bae_robodiff/bin/python
$PY -c "import timm, torch; print('timm ok', torch.cuda.is_available())"

# (b) SSD 마운트 + slow(base) ckpt 존재 확인
export RESIDUAL_SLOW_CKPT=/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt
ls -la "$RESIDUAL_SLOW_CKPT"    # 없으면 SSD 다시 마운트 (경로는 UUID 폴더)

# (c) workdir — actor/learner 공유. 낡은 폴더 재사용 금지(차원/버전 충돌).
export RESIDUAL_ONLINE_WORKDIR=data/online_runs/run_hand_residual
rm -rf "$RESIDUAL_ONLINE_WORKDIR"   # 새 실험이면 비우고 시작

# (d) 로봇 없이 코어 sanity (SSD만 있으면 됨)
$PY online_learning/smoke_test_residual_no_robot.py   # "🎉 전부 통과" 떠야 함
```

체크: `smoke`가 통과하면 learner/dataset/relabel/hot-swap은 정상. 이제 actor만 남는다.

---

## 1. Actor — 초안 작성됨, 로봇에서 [VERIFY] 확인만

`online_learning/residual_teleop_actor_env_runner.py` 가 이미 작성돼 있다(one-step-per-tick,
head hot-swap, slow_pred 로깅, residual 전송, Δ캡, 교정 핸드오프 포함). `py_compile` 통과.
**로봇에서 아래 [VERIFY] 만 실측 확인·튜닝하면 바로 돈다:**

- **[VERIFY-1] 이미지 스케일** (`_to_uint8_image`): env(obs_float32=True) 의 `image0` 이 0~1 인지
  0~255 인지 확인. 코드가 max≤1.5 면 ×255 로 추정하지만, 실제 스케일을 한 프레임 찍어보고
  맞는지 확인(학습 이미지와 스케일이 다르면 정책이 엉뚱해짐).
- **[VERIFY-2] Δ캡** (`--residual_translation_cap 0.05 --residual_rotation_cap 0.4`): 초기 head
  발산 시 로봇이 안 튀는지. 작게 시작해 키워라.
- **[VERIFY-3] 교정 중 slow 주기**: teleop 모드는 매 tick slow 재추론(신선한 base)이라 ~5Hz 로
  느려질 수 있음. 느리면 `--num_inference_steps` 낮추거나 교정을 짧게.
- **[VERIFY-4] exec_actions 정합**: 정책 모드 1스텝 실행 timestamp(`obs_ts[-1]+dt`)가 로봇
  보간과 맞는지(너무 촘촘/성기면 `--frequency` 조정).
- **[VERIFY-5] servo 핸드오프**: `a`→pause+teleop=1, `b`→teleop=4+resume 가 servo 노드와
  안 싸우는지(online actor 와 동일 로직이지만 재확인).

아래는 그 actor 가 **어떻게 구현됐는지**(참고). 구조를 바꿔야 할 때만 본다.

### 구현 근거 (참고용) — 두 검증된 코드의 병합
- 추론 골격: `diffusion_policy/residual_policy/eval_real_robot_rightarm_insert_plug.py`
  - slow가 chunk 예측 → per-step fast Δ refine → `apply_residual_action_to_pose9` → 1스텝 실행.
  - 실행 직전 `slow_abs_target`(pose9)가 **그 스텝의 base**. (파일 내 line ~776, `final_abs_action` 계산 직전)
  - 선택: `--slow_use_pigdm` realtime chunking(떨림 완화, 나중에).
- 온라인/교정 골격: `online_learning/finetune_teleop_actor_env_runner.py`
  - a/b/c servo 핸드오프, teleop correction 기록, `KeyReader`, `FileMailbox`, 에피소드 전송, `use_hand`.

권장: `online_learning/residual_teleop_actor_env_runner.py` 를 새로 만들되 위 둘에서 함수 복붙.

### 얹을 온라인 훅 3개 (정확히 이것만 추가)

**훅 1 — head hot-swap (에피소드 경계에서).** full state_dict 로드( `maybe_hotswap_weights` ) 대신:
```python
def maybe_hotswap_residual_head(mailbox, fast_policy, current_version, device):
    latest = mailbox.get_latest_weight_version()
    if latest is None or latest == current_version:
        return current_version
    payload = mailbox.load_weights(latest, map_location=device)
    if payload is None or "head_state" not in payload:
        return current_version
    fast_policy.head.load_state_dict({k: v.to(device) for k, v in payload["head_state"].items()})
    fast_policy.normalizer.load_state_dict(
        {k: v.to(device) for k, v in payload["normalizer_state"].items()}, strict=False)
    if payload.get("force_encoder_state") is not None and fast_policy.force_encoder is not None:
        fast_policy.force_encoder.load_state_dict(
            {k: v.to(device) for k, v in payload["force_encoder_state"].items()})
    fast_policy.eval().to(device)
    print(f"[Actor] head hot-swap v{current_version} -> v{latest} (demo={payload.get('num_demos','?')})")
    return latest
```
슬로우(frozen)는 절대 안 건드림. actor는 slow ckpt를 자기 쪽에서 이미 로드해 갖고 있음(learner와 동일 경로).

**훅 2 — slow_pred 로깅 (실행하는 스텝마다).** 실행 직전 `slow_abs_target`(pose9)를 에피소드 리스트에 append. replay_buffer에 기록되는 achieved obs 프레임과 **같은 인덱스**로 쌓을 것.
```python
slow_pred_log = []          # 에피소드 시작 시 초기화
...
slow_pred_log.append(np.asarray(slow_abs_target, dtype=np.float32).copy())   # 매 실행 스텝
```

**훅 3 — relabel + 전송 (에피소드 유지 시).** replay_buffer의 achieved obs + 훅2 로그로 dict 구성 → 기존 relabel util 사용:
```python
from online_learning.residual_relabel_utils import write_residual_episode_hdf5
# replay_buffer 마지막 에피소드에서 per-step 배열 추출 (RAW_OBS_KEYS 순서·길이 동일해야 함)
ep = {
    "image0": <T,H,W,3 uint8>,           # obs_res 로 리사이즈된 카메라
    "robot_pose_R": <T,3>, "robot_quat_R": <T,4>,
    "hand_pose_R": <T,7>, "wrench_wrist_R": <T,6,32>,
    "slow_pred_target_abs": np.stack(slow_pred_log, 0),   # <T,9>  ★ 길이 = 위 obs 길이와 동일
}
out = write_residual_episode_hdf5(os.path.join(WORKDIR, "last_episode.hdf5"), ep, demo_name="demo_0")
mailbox.send_episode(out)
```
relabel이 내부에서 `virtual[t] = achieved(t+1)`, `residual_delta6[t] = Δ(slow_pred[t] → virtual[t])`를
계산하고 마지막 스텝을 버린다(T→T-1). **그래서 task config는 `action_target_shift=0`**.

### 정렬 함정 (여기서 대부분 실수함)
- `slow_pred_log[t]` 는 반드시 **그 스텝 t 에서 실제 로봇에 실행한 base**(=slow_abs_target)여야 한다.
  chunk를 여러 스텝 실행하면, 실행한 각 스텝마다의 slow_abs_target을 그 스텝 obs와 짝지어 append.
- eval 스크립트의 `slow_action_start_offset`/`fast_action_target_shift`는 slow chunk **내부** 인덱싱일 뿐,
  온라인 relabel의 정렬(shift=0)과 무관. 헷갈리지 말 것.
- 길이 불일치 방지: `len(slow_pred_log) == len(replay_buffer 해당 에피소드 프레임)` 을 assert.

### 안전 (실로봇 필수)
- **Δ 캡**: 실행 전 residual delta 상한 — 병진 5cm, 회전 0.4rad (cr-dagger `scale_and_cap_residual_action`).
  초기 head가 발산해도 로봇이 안 튀게. `apply_residual_action_to_pose9` 직전에 residual을 clip.
- **임피던스 stiffness 낮게** — 사람이 손으로 밀 수 있을 만큼(병진 ~1000 N/m 기준).
- **손은 gate-teleop relative** 유지(manus 기존 설정, `manus_to_aidin_rush.py --gate-teleop -r --relative`).
- 사람 손은 **카메라 시야 밖**에서 밀 것(시야 안이면 policy가 "손 보이면" 오학습).

---

## 2. 실행 (멀티터미널)

full-finetune 경로와 동일한 4터미널 구조. **모든 터미널에서 아래 3개 export를 먼저** 하고 시작.
```bash
PY=/home/rush/anaconda3/envs/bae_robodiff/bin/python
export RESIDUAL_SLOW_CKPT=/media/rush/.../260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt
export RESIDUAL_ONLINE_WORKDIR=data/online_runs/run_hand_residual
```

| 터미널 | 명령 | 역할 |
|--------|------|------|
| 1 (servo) | `$PY online_learning/servo_rightarm_imp_online.py` | 팔 servo(VR Vive 트래커 + `/teleop_control=1`) |
| 2 (learner) | `$PY online_learning/residual_teleop_learner.py` | head warm-continue 학습 + 발행 |
| 3 (manus) | `(manus_ws)$ python manus_to_aidin_rush.py --gate-teleop -r --relative` | 손 교정(교정 중에만 발행) |
| 4 (actor) | `$PY online_learning/residual_teleop_actor_env_runner.py -i "$RESIDUAL_SLOW_CKPT" --use_hand --steps_per_inference 6 --frequency 10 --num_inference_steps 12` | slow+fast 추론 + 교정 수집 + hot-swap + 전송 |

actor 키: `a`=servo 핸드오프(교정 시작) · `b`=actor 복귀 · `c`=홈 · `s`=유지+전송 · `d`=폐기 · `q`=종료.
(actor 터미널에 포커스 두고 페달/키 입력.)

### 운영 루프 (사람이 하는 일)
1. actor가 slow+residual로 task 수행하는 걸 지켜봄.
2. 어긋나면 `a`(servo 핸드오프) → 팔을 물리적으로 밀어 교정(+필요시 manus로 손) → `b`로 복귀.
3. `s`로 에피소드 유지(자동 relabel 후 learner 전송) 또는 `d`로 폐기.
4. learner가 head 갱신·발행(라운드 수초) → **다음 에피소드 시작 시 actor가 head hot-swap**.
5. 반복하며 head가 사람 교정을 점점 반영.

---

## 3. 튜닝 노브 (환경변수, learner)
| 변수 | 기본 | 의미 | 로봇 권장 |
|------|------|------|-----------|
| `RESIDUAL_LR` | 1e-4 | head 학습률 | 5e-5~2e-4 |
| `RESIDUAL_FIRST_EPOCHS` | 120 | 첫 라운드 부트스트랩 | 80~150 |
| `RESIDUAL_EPOCHS_PER_ROUND` | 40 | 이후 라운드 | 20~50 |
| `RESIDUAL_MIN_EPISODES` | 2 | 첫 학습 시점 | 2~4 |
| `RESIDUAL_MAX_SAMPLES_PER_EPOCH` | 0(전체) | 라운드 속도 상한 | 데이터 많아지면 256~512 |
| `RESIDUAL_BATCH_SIZE` | 64 | | |

actor 노브: `--steps_per_inference`(slow 재계획 주기, 작을수록 반응↑·연산↑), `--num_inference_steps`(DDIM),
`--slow_use_pigdm`(chunk 떨림 완화).

---

## 4. 검증 (실행 중 확인)
1. **learner loss↓**: 터미널2에서 라운드마다 `loss=...` 감소, `가중치 발행: vN` 로그.
2. **hot-swap 반영**: 터미널4에서 `head hot-swap v.. -> v..` 뜨고 정책 거동 변화.
3. **망각 없음(핵심 이점)**: frozen base라 **옛 위치도 계속 성공**해야 함. full-finetune과 달리
   새 위치만 되고 옛 위치 깨지는 현상이 없어야 정상. 옛/새 위치 번갈아 테스트.
4. **wrench 기여**(선택): `analysis/modality_attribution/batch_build_viewers` 로 Δwrench 재측정.
   (train_force_encoder=True로 켠 v2와 비교.)

---

## 5. 트러블슈팅
| 증상 | 원인/해결 |
|------|-----------|
| `ModuleNotFoundError: timm` | `robodiff` env로 돌림 → `bae_robodiff` 사용. |
| slow ckpt 없음 | SSD 미마운트. §0(b) 경로 확인, 다시 마운트. |
| hot-swap "shape mismatch"/스킵 | 낡은 workdir 재사용. `RESIDUAL_ONLINE_WORKDIR` 새로 비우고 learner·actor 재시작. |
| actor 로봇이 튐 | Δ 캡 미적용(§1 안전) 또는 stiffness 과다. residual clip 넣고 stiffness↓. |
| relabel 길이 assert 실패 | `slow_pred_log` 와 obs 프레임 인덱스 어긋남(훅2). 실행 스텝마다 1:1로 쌓았는지 확인. |
| 정책이 손 교정 반영 안 함 | v1은 **pose-only 6D**라 손 residual 학습 안 함(설계). 손 교정 학습은 v2(HANDOFF §6). |
| learner 재시작 시 head v0부터 | 정상(누적 초기화). 보존하려면 실행 전 workdir 백업. |

---

## 6. 완료 정의 (이번 목표) — 남은 체크리스트
- [x] learner 코어(config·relabel·learner) 작성 + 로봇없이 smoke 통과
- [x] actor 초안 작성(훅3 + Δ캡, py_compile OK)
- [ ] **actor [VERIFY] 1~5** (§1) 로봇에서 실측 확인
- [ ] 4터미널 기동 (§2)
- [ ] 교정 몇 에피소드 → learner loss↓ & head vN 발행
- [ ] hot-swap 후 정책 거동 변화 관찰
- [ ] 옛/새 위치 성공률 동시 확인(망각 없음 = residual-online의 핵심 검증)
- [ ] (v2, 선택) 손 residual / wrench→virtual-target compliance / correction 가중 샘플링 / RTC
      — 상세는 `RESIDUAL_ONLINE_HANDOFF.md` §6
