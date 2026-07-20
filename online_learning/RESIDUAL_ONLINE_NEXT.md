# ⭐ 최우선 (P0): 로봇 재수집 → residual 재분석

> **지금 가장 먼저 할 일.** 로봇에서 페달로 교정 데이터를 새로 쌓고(→ residual head 온라인 학습),
> 그 데이터를 검증 파이프라인/3D HTML로 다시 분석한다.
> 배경/설계는 `RESIDUAL_ONLINE_HANDOFF.md`, 기동은 `RESIDUAL_ONLINE_QUICKSTART.md`(vision 머신),
> 상세 절차/튜닝은 `RESIDUAL_ONLINE_ROBOT_RUNBOOK.md`.

## 0. 검증 완료된 사실 (2026-07-20)
라이브 actor가 `mailbox.send_episode`로 내보내는 **residual-포맷 에피소드**를
분석 스크립트(`replay_feed_episodes` / `verify_residual_on_data` / `offline_replay_verify`)가
**그대로 읽는다**(실측 확인: RAW_OBS_KEYS 존재, image0=uint8 0~255, slow_pred 재계산 OK).
→ 즉 **페달 밟고 `s`로 send만 하면, 쌓인 걸 재분석 가능**하다. 아래 4가지만 지키면 "의미 있는" 분석이 된다.

## 1. 흐름 (페달 → send → 쌓임 → 분석)
```
4_actor.sh  → 페달 교정 → s(유지) → write_residual_episode_hdf5 → mailbox
             → <WORKDIR>/transitions/ep_*.hdf5      ← 여기 쌓임 (분석 소스)
2_learner.sh → transitions 폴링 → warm-continue 학습 → head 발행(<WORKDIR>/weights)
수집 후      → 분석 스크립트가 transitions/*.hdf5 를 읽어 재분석 + 3D HTML
```
공통 변수(launch_abs 기준): `WORKDIR=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual_abs`,
`RESIDUAL_SLOW_CKPT`=actor·learner가 쓴 **바로 그 abs ckpt**(260710), env=`venv_diffusion`.

## 2. ⚠️ 잘 분석되려면 반드시 지킬 4가지
1. **held-out을 남겨라 (가장 중요).** 모든 에피소드가 learner에 들어가면 분석은 "학습셋 재현"만 보여줌(낙관적).
   → 수집한 `transitions/*.hdf5`를 **`transitions_train/`(학습) + `transitions_heldout/`(검증) 두 폴더로 나눠** 둘 것.
   일반화 판단은 반드시 held-out으로.
2. **10개 이상 수집(train 7~8 + heldout 2~3).** 3개짜리 테스트는 미학습이라 하드프레임 지표가 음수였고,
   run_hand 38개로는 방향 cos 0.89가 나왔다. "몇 번"으론 부족할 수 있음.
3. **[VERIFY-1] 이미지 스케일.** 실로봇 카메라(obs_float32)에서 actor의 `_to_uint8_image`가 0~255 uint8을
   제대로 만들어야 함. 틀리면 **라이브 residual도, 분석 slow 재계산도 둘 다 조용히 망가짐.** 첫 프레임 1장 찍어 확인.
4. **base/config/workdir 일치.** actor=260710 abs(launch_abs) → 분석도 `--config_name residual_policy/hand_online_abs_mlp`
   + **같은 abs ckpt**. 새 실험이면 **workdir 비우고 시작**(낡은 head hot-swap 방지).

## 3. 실행 스텝
```bash
PY=/home/vision/venv_diffusion/bin/python
WD=/home/vision/rush_diffusion_policy/data/online_runs/run_hand_residual_abs
export RESIDUAL_SLOW_CKPT=/home/vision/rush_diffusion_policy/data/outputs/260710_insert_box_hand_wrench_abs/epoch=0900-train_loss=0.001.ckpt
export RESIDUAL_ONLINE_WORKDIR=$WD
cd /home/vision/rush_diffusion_policy

# (0) 새 실험이면 workdir 비우기
rm -rf $WD

# (1) 수집: launch_abs 4터미널 (servo/learner/manus/actor). 페달로 교정 → s로 send. 10+ 에피소드.
#     online_learning/launch_abs/{1_servo,2_learner,3_manus,4_actor}.sh
#     ※ 첫 프레임에서 [VERIFY-1] 이미지 스케일 확인!

# (2) 수집 끝나면 held-out 분리 (예: 마지막 3개를 held-out으로)
mkdir -p $WD/transitions_train $WD/transitions_heldout
ls $WD/transitions/ep_*.hdf5 | head -n -3 | xargs -I{} cp {} $WD/transitions_train/
ls $WD/transitions/ep_*.hdf5 | tail -n 3      | xargs -I{} cp {} $WD/transitions_heldout/

# (3-A) 처음부터 재학습 + held-out 검증 + 그림 (권장, 자기완결)
$PY online_learning/offline_replay_verify.py \
  --train_source $WD/transitions_train --heldout_source $WD/transitions_heldout \
  --out data/verify_new
#   → data/verify_new/{verify_report.json, verify_plots.png}

# (3-B) 또는 라이브 학습된 head 그대로 held-out 검증만
$PY online_learning/verify_residual_on_data.py \
  --slow_ckpt $RESIDUAL_SLOW_CKPT --config_name residual_policy/hand_online_abs_mlp \
  --heldout $WD/transitions_heldout --out data/verify_new
```
3D HTML은 `data/residual_verify_abs/traj3d.json` 추출 방식(이 세션에서 쓴 export 스니펫)으로 재생성 →
아티팩트 재발행. (traj3d 추출 스크립트를 정식 파일로 뽑아둘지는 아래 TODO.)

## 4. 결과 해석
- **train(같은 데이터)는 낙관적** — residual이 achieved에 거의 붙는 게 당연(암기). 일반화는 **held-out**으로.
- 핵심 지표: 교정 큰 상위25%(=잘못된 상황)에서 **방향 cos ↑, 개선 프레임 % ↑, 교정 포착률 ↑** 이면 "교정 쪽으로 되돌린다"(#4).
- **세션 간 수렴 신호**: 새 에피소드의 `achieved − slow` = (head residual) + (남은 사람교정). 세션 거듭할수록 이 값이 줄면 head가 교정을 흡수 중 = 진짜 개선.
- 주의: run_hand 실측상 교정은 **90%가 base에서 2.6cm 이내(작은 국소 보정), 상위 5%만 5~12cm(큰 복구)**.
  그 큰 복구가 **Δ캡 5cm에 잘리고 OOD 위험 최고** — 실패는 거기서 난다. (실패모드 상세는 대화 기록/HANDOFF 참고.)

## 5. 관련 파일 (git으로 가져갈 것)
- 분석: `online_learning/{replay_feed_episodes,verify_residual_on_data,offline_replay_verify}.py`
- abs config: `diffusion_policy/config/residual_policy/{hand_online_abs_mlp.yaml, task/hand_online_abs.yaml}`
- 코어(기존): `online_learning/{residual_online_learner,residual_relabel_utils,config_residual_online}.py`,
  `residual_online_actor_env_runner.py`, `launch_abs/*.sh`

## 6. 선택 개선 TODO (하면 편해짐, 필수 아님)
- `--heldout_frac 0.25` 자동 분할(폴더 수동 분리 제거).
- `--use_logged_slow_pred` : actor가 로깅한 실제 slow_pred 사용(재계산 대신 → 로봇이 실제 쓴 base와 정확 일치 + 빠름).
- traj3d 추출을 정식 스크립트(`export_traj3d.py`)로 승격 + 3D HTML 자동 재발행.
