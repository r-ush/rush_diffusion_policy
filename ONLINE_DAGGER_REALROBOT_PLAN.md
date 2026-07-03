# 실로봇(오른팔) 온라인 DAgger 마무리 구현 계획

> **용도**: 이 문서는 구현 지시서다. 지금까지 로봇 코드를 몰라 보류했던 부분을 실제 코드
> (`rush_eval_real_robot_imp_right.py`, `rush_real_env_rightarm_imp.py`,
> `rightarm_interpolation_controller_imp.py`)를 기준으로 확인 완료했고, 남은 구현을
> 아래 순서대로 진행하면 된다. 전체 시스템 배경은 `ONLINE_DAGGER_HANDOFF.md` 참고.
>
> 페달 입력은 **이번 범위에서 제외** (키보드 C/S/D 유지, 페달은 나중에 키 에뮬레이션으로 매핑).

---

## ✅ 구현 상태 (2026-07-03 완료)

이 문서의 §1~§5를 모두 구현했다. 로봇 없이 가능한 검증은 통과.

| # | 작업 | 결과 |
|---|------|------|
| 1 | `relabel_utils.py` 영상 디코드(`load_episode_frames`, read_video 재사용) + `relabel_last_episode_to_hdf5(replay_buffer, env_output_dir, ...)` 시그니처 변경 | ✅ 구현·임포트 OK |
| 2 | `rush_replay_buffer_to_correction_hdf5.py` — `--input`을 env output_dir로, 영상 디코드 사용 | ✅ 구현·임포트 OK |
| 3 | actor를 `RightarmRealEnvImp` 기반으로 재작성 (hf_hub shim / click CLI / pose_repr 분기 / steps_per_inference 절단 / max_duration / 새 relabel 시그니처 / hot-swap) | ✅ 구현·컴파일 OK (로봇 실행만 미검증) |
| 4 | `config_online.py` env override(`ONLINE_BASE_CKPT`/`ONLINE_BASE_DATASET`/`ONLINE_NUM_BASE_DEMOS`) + 실로봇 권장 기본값 / `online_learner.py --input` | ✅ 구현 OK |
| 5 | `rush_eval_real_robot_imp_right.py`에 C/S/D + stages 기록 | ✅ 구현·컴파일 OK |

**검증 내역**
- 전 파일 `py_compile` 통과.
- `smoke_test_no_robot.py` 전체 통과 (전송→학습 라운드→v1 발행→hot-swap(missing=0/unexpected=0)→predict_action).
  - 단, 이 개발 머신엔 `epoch=0700` ckpt와 `..._des.hdf5`가 더 이상 없어
    `ONLINE_BASE_CKPT=.../checkpoints/latest.ckpt`, `ONLINE_BASE_DATASET=.../20260630_195919_diffusion_des_10.hdf5`
    로 override해 실행함. **로봇 PC에서도 실제 ckpt/데이터셋 경로를 -i / 환경변수로 지정할 것**
    (config 기본값은 이 머신 기준의 stale 경로).
- **영상 디코드 정렬 검증**: env와 동일한 `VideoRecorder` 경로로 30프레임 합성 mp4를 만들고
  `load_episode_frames`로 디코드 → 30/30 프레임, 순서(밝기 단조성) 일치 확인. 즉 frame k ↔
  obs step k 1:1 정렬이 실제로 성립함.

**로봇에서 남은 검증** (§6 체크리스트): 실제 카메라·팔에서 actor 실행, relabel 이미지가
카메라와 색상/정렬 맞는지(§1-3), correction stage 기록, hot-swap 후 거동.

---

## 0. 실코드 확인 결과 (이번에 새로 파악된 사실)

실제 로봇 inference 코드는 **오른팔** `rush_eval_real_robot_imp_right.py`이며
(`run_right.sh`: ckpt=`data/outputs/logistics_abs_10/epoch=0800-train_loss=0.000.ckpt`,
`--steps_per_inference 12 --frequency 10 --num_inference_steps 12`),
env는 `diffusion_policy/real_world/rush_real_env_rightarm_imp.py`(`RightarmRealEnvImp`)다.

### 이미 준비돼 있어서 수정 불필요한 것 ✅
- `RightarmRealEnvImp.exec_actions(..., stages=)` — correction 플래그(`stage`) 기록 지원됨
  (`rush_real_env_rightarm_imp.py:327-365`).
- `RightarmRealEnvImp.drop_episode()` — 에피소드 폐기 지원됨 (`:447`).
- obs key가 오른팔이어도 `robot_pose_L`/`robot_quat_L` 라벨 유지 → 기존 relabel/learner의
  key 이름과 그대로 호환.
- 컨트롤러 `schedule_waypoint(pose)`가 9D(pos3+rot6d) 입력을 받음
  (`rightarm_interpolation_controller_imp.py:238`).
- `VideoRecorder.write_frame`에 float→uint8 변환 패치가 이미 있음
  (`video_recorder.py:144`, `img*255` clip) → `obs_float32=True`여도 영상 정상 저장.
- 영상이 `record_raw_video=False` 경로로 **obs 해상도(320×240) RGB, 제어주기(10fps)**로
  저장되고, `write_frame`이 `get_accumulate_timestamp_idxs(start_time, dt=1/fps)` 그리드로
  기록함 → **영상 frame k ↔ 에피소드 step k가 1:1 인덱스 정렬** (relabel에 그대로 사용 가능).

### 보류했다가 이제 확정된 문제 (구현 대상) ⚠️
1. **[Blocker] replay_buffer.zarr에 이미지가 없다.**
   `obs_accumulator`는 로봇 lowdim 키만 누적한다. zarr에는
   `robot_pose_L, robot_quat_L, action, stage, timestamp`만 있고, 이미지는
   `<output_dir>/videos/{episode_id}/{cam_idx}.mp4` 에만 있다.
   그런데 `online_learning/relabel_utils.py`와
   `data_process/rush_replay_buffer_to_correction_hdf5.py`는 둘 다
   `replay_buffer['image0']`를 가정하고 있어 **실로봇 데이터에서 그대로 죽는다.**
2. **actor가 왼팔 env 기반이다.** `online_actor_env_runner.py`가 `LeftarmRealEnvImp`를
   import하고, 오른팔 실코드에 있는 relative pose_repr 분기 / steps_per_inference 절단 /
   max_duration 타임아웃 / huggingface_hub sys.path shim / CLI 옵션이 없다.
3. **config가 이 머신 기준이다.** `BASE_CKPT`가 왼팔 ckpt 하드코딩(환경변수 override 불가),
   `BASE_DATASET_PATH`가 `/home/rush/Desktop/...` (로봇 PC에는 없음).
   로봇 PC는 `/home/vision/miniconda3/envs/robodiff` (shebang 기준) — 사용자/경로가 다르다.

---

## 1. [최우선] relabel 이미지 소스를 영상 디코드로 교체

### 1-1. `online_learning/relabel_utils.py`
- 함수 추가:
  ```python
  def load_episode_frames(env_output_dir, ep_index, n_steps, cam_idx=0):
      """videos/{ep_index}/{cam_idx}.mp4 를 디코드해 (T,H,W,3) uint8 RGB 반환.
      frame k ↔ step k 1:1 정렬 (VideoRecorder가 dt 그리드로 기록하므로).
      len(frames) != n_steps 이면 min으로 clip하고, 차이가 2프레임 초과면 경고 출력."""
  ```
  - `cv2.VideoCapture` 사용 시 **BGR→RGB 변환 필수** (영상은 rgb24로 인코딩됐지만
    cv2가 BGR로 돌려줌). 또는 `av`로 디코드(`video_recorder.read_video` 참고).
  - 영상은 이미 320×240이므로 resize 불필요. 단, 해상도가 다르면
    `get_image_transform`으로 맞추는 방어 코드.
- `_extract_episode_arrays()`에서 `replay_buffer['image0']` 제거 → lowdim만 추출.
- `relabel_last_episode_to_hdf5(replay_buffer, out_path)` 시그니처 변경:
  `relabel_last_episode_to_hdf5(replay_buffer, env_output_dir, out_path)` —
  내부에서 `ep_index = n_episodes-1`, lowdim 추출 후 `load_episode_frames(...)`로
  이미지 획득. **이미지는 uint8 그대로 HDF5에 저장** (float 변환 분기 제거 가능).
- `smoke_test_no_robot.py`와 GUI 데모(`temp_project_sim_pushtest/online_loop_demo.py`)는
  배열을 직접 넘기는 `relabel_episode_to_hdf5(image0, pose, quat, out_path)`를 쓰므로
  이 함수는 시그니처 유지 → 데모/스모크는 깨지지 않게 한다.

### 1-2. `data_process/rush_replay_buffer_to_correction_hdf5.py` (오프라인 경로)
- 동일하게 zarr `image0` 가정 제거. `--input`을 replay_buffer.zarr가 아닌
  **env output_dir**(replay_buffer.zarr + videos/ 를 포함한 폴더)로 받도록 변경하고
  에피소드별로 `load_episode_frames` 사용.
- 대안(선택): `diffusion_policy/real_world/real_data_conversion.py`의
  `real_data_to_replay_buffer(dataset_path, image_keys=['camera_0'], ...)` 재사용.
  단 전 에피소드 일괄 변환이라 온라인 경로에는 부적합 — 온라인은 마지막 에피소드만
  디코드하는 1-1 방식을 쓴다.

### 1-3. 검증 방법
로봇에서 짧은 에피소드 1개 기록 → relabel HDF5의 `obs/image0[k]`와
`videos/{ep}/0.mp4`의 frame k를 나란히 시각화해 정렬/색상(RGB) 확인.
`robot_pose_L[k]` vs `actions[k]`(=step k+1 achieved pose)가 한 스텝 시프트인지 확인.

---

## 2. actor를 오른팔 실코드 기준으로 재작성

`online_learning/online_actor_env_runner.py`를 `rush_eval_real_robot_imp_right.py`의
제어 루프와 **동일 구조**로 맞춘다 (검증된 실코드에서 벗어나지 않는 것이 원칙).

수정 목록:
1. **huggingface_hub sys.path shim**을 파일 최상단에 추가
   (오른팔 eval 파일 26-36행과 동일 — 로봇 PC conda 환경 충돌 회피용, 다른 import보다 먼저).
2. `LeftarmRealEnvImp` → `RightarmRealEnvImp` (`rush_real_env_rightarm_imp`).
3. **click CLI 추가** (오른팔 eval과 동일 옵션 + 온라인 전용):
   - `--input/-i` ckpt 경로 (기본값 `C.BASE_CKPT`) — 로봇에선 logistics_abs_10 ckpt를 넘김
   - `--output/-o` (기본값 `C.ACTOR_OUTPUT_DIR`)
   - `--steps_per_inference 12`, `--frequency 10`, `--num_inference_steps 12`,
     `--max_duration 60`
4. **pose_repr 분기** (오른팔 eval 127-135, 171-176, 213-218, 256-278행 미러링):
   - `obs_pose_repr=='relative'`면 `get_real_relative_obs_dict`, 아니면 `get_real_obs_dict`
   - `action_pose_repr=='relative'`면 `get_abs_action_from_relative`로 abs 변환 후 실행
   - warmup inference에도 동일 분기 적용
   - (logistics_abs_10은 abs라 당장은 abs 경로만 타지만, 분기는 cfg에서 자동 결정되게)
5. **타이밍 처리 오른팔 버전으로**: `this_target_poses[is_new][:steps_per_inference]`
   절단 + over-budget 시 마지막 action fallback + `max_duration` 타임아웃(현재 actor에 없음).
6. 키 입력은 현행 유지: `C` correction 토글, `S` 유지+전송, `D` 폐기.
   타임아웃 종료 시에도 `keep=True`로 전송할지 여부는 **전송한다**로 구현
   (성공적으로 끝난 정상 에피소드도 학습 데이터로 유효 — correction 없으면 self-BC라 무해).
7. relabel 호출부를 새 시그니처로:
   `relabel_last_episode_to_hdf5(env.replay_buffer, env_output_dir=output, out_path=...)`.
   전송 파일명은 덮어쓰기 경쟁 방지 위해 `last_episode.hdf5` 대신
   `ep_{episode_id}.hdf5` 권장 (mailbox.send_episode가 어차피 복사한다면 현행 유지 가능 —
   mailbox 구현 확인 후 결정).
8. hot-swap(`maybe_hotswap_weights`)은 현행 유지 — 에피소드 경계에서만.
9. 오른팔 eval의 orientation 디버그 블록은 try/except라 그대로 가져와도 무해 (선택).

---

## 3. `online_learning/config_online.py` 보강

1. `BASE_CKPT`에 환경변수 override 추가:
   `BASE_CKPT = os.environ.get("ONLINE_BASE_CKPT", <기존 기본값>)`.
   (actor는 `-i`로도 덮어씀. **learner도 같은 ckpt에서 시작해야 하므로** learner 실행 전
   `ONLINE_BASE_CKPT`를 export하거나 learner에도 `--input` CLI를 추가 — 후자 권장.)
2. `BASE_DATASET_PATH`도 `ONLINE_BASE_DATASET` 환경변수 override 추가.
   로봇 PC에서 오른팔 task의 base 학습 HDF5 경로로 지정해야 함 (사용자 제공 필요).
3. 실로봇 권장값 반영: `NUM_BASE_DEMOS_TO_MIX=10`(기본 0 유지하되 주석으로 강조),
   `MIN_EPISODES_BEFORE_TRAIN=3`, `MAX_SAMPLES_PER_EPOCH=256`.
4. 주의 주석 추가: 로봇 PC는 `/home/vision/...`이므로 절대경로 금지, ROOT 상대경로 유지.
   `data/`는 gitignore라 **ckpt와 base HDF5는 수동 복사** 필요.

---

## 4. learner 확인 사항 (코드 수정은 조건부)

1. **actor와 learner의 ckpt 일치**: learner도 `C.BASE_CKPT`를 로드하므로 3-1의
   override가 반영되면 됨. learner에 `--input` CLI를 추가하면 실수 여지가 줄어든다.
2. **relative-action ckpt 대비 (logistics_abs_10은 abs라 지금은 해당 없음)**:
   relabel HDF5는 abs achieved pose를 저장한다. ckpt의 `cfg.task.pose_repr`이 relative면
   learner의 `cfg.task.dataset`(BaeRobomimicReplayDataset)이 abs→relative 변환을
   해주는지 확인하고, 안 해주면 relabel 단계에서 변환 추가. **구현 시
   `cfg.task.pose_repr`을 print해서 abs면 이 항목은 스킵.**
3. 누적셋 이미지가 uint8로 저장되므로 dataset이 uint8 image0을 정상 정규화하는지
   (base HDF5도 uint8이므로 동일 경로 — 문제 없을 것으로 예상, 첫 라운드 loss로 확인).

---

## 5. (선택) `rush_eval_real_robot_imp_right.py`에 C/S/D 추가

온라인과 별개로 **오프라인 수집→재학습 경로**도 오른팔에서 쓰려면, 왼팔
`rush_eval_real_robot_imp.py`에 했던 것과 동일한 패치를 오른팔 파일에도 적용:
- 루프 진입 전 `correction_active=False`
- `cv2.pollKey()`를 exec_actions **앞**으로 이동, `C` 토글 처리
- `env.exec_actions(..., stages=np.full(len(poses), 1 if correction_active else 0))`
- `S`=end_episode(유지), `D`=drop_episode(폐기)
온라인 actor(§2)가 이 기능을 포함하므로, 온라인만 쓸 거면 이 항목은 생략 가능.

---

## 6. 로봇 반입 체크리스트 & 검증 순서

```
[반입] 리포 디렉토리 통째 복사 + 수동 복사: ckpt(logistics_abs_10), base HDF5(mix용)
[0] conda activate robodiff; python online_learning/smoke_test_no_robot.py   # 통신/학습 루프
[1] actor 단독, SEND_TRANSITIONS=False로 순수 inference 확인 (기존 run_right.sh와 동일 거동인지)
[2] 짧은 에피소드 1개 → §1-3 relabel 검증 (이미지 정렬/RGB/한스텝 시프트)
[3] learner 기동 → 에피소드 전송 → 학습 라운드 → v1 발행 → 다음 에피소드에서 hot-swap 로그 확인
[4] correction 시나리오: C 토글하며 팔 밀기 → stage=1 구간이 zarr에 기록되는지 확인
[5] 운영: ONLINE_DAGGER_HANDOFF.md §3-5 루프대로
```

안전: 첫 hot-swap 후 policy가 튀는지 감시. 필요 시 relabel 단계에서 스텝 간 pose delta
상한(병진 5cm/스텝) 캡 추가 — HANDOFF §6 참고 (이번 범위엔 선택).

---

## 7. 구현 순서 요약 (Opus 작업 리스트)

| # | 작업 | 파일 | 난이도 |
|---|------|------|--------|
| 1 | 영상 디코드 relabel (`load_episode_frames`) | `online_learning/relabel_utils.py` | 중 |
| 2 | 오프라인 스크립트 동일 수정 | `data_process/rush_replay_buffer_to_correction_hdf5.py` | 하 |
| 3 | actor 오른팔 재작성 (shim/CLI/pose_repr/타임아웃/절단) | `online_learning/online_actor_env_runner.py` | 중 |
| 4 | config env override + learner `--input` | `config_online.py`, `online_learner.py` | 하 |
| 5 | (선택) 오른팔 eval에 C/S/D | `rush_eval_real_robot_imp_right.py` | 하 |
| 6 | 스모크/데모 회귀 확인 (`smoke_test_no_robot.py`, GUI 데모) | — | 하 |
| 7 | HANDOFF 문서에 오른팔 기준 반영 | `ONLINE_DAGGER_HANDOFF.md` | 하 |

구현 후 이 문서의 §6 체크리스트를 로봇에서 그대로 실행하면 된다.
