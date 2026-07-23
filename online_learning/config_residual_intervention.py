"""
온라인 residual DAgger(개입형, physical push) 공유 설정 — actor & intervention learner.

config_residual_online.py(teleop 판) 대응. **속성 이름은 동일**하게 두어 공유 learner
(residual_teleop_learner.ResidualOnlineLearner)가 `self.C` 로 이 모듈을 그대로 받아 쓸 수
있게 한다. teleop 판과 차이:
  * WORKDIR / config_name 기본값이 개입 전용(run_hand_intervention / hand_intervention_mlp).
  * 개입 프레임 가중 샘플링용 INTERVENTION_SAMPLE_WEIGHT 추가.

개입형 vs teleop 판(개념):
  * teleop: 페달로 servo/manus 에 제어를 넘겨 사람이 팔+손을 모두 몬다.
  * 개입형: 제어를 넘기지 않는다. 팔은 base(+residual)가 임피던스로 계속 나가고, 손도 base 가
    자율 제어하며, 사람은 그 팔을 물리적으로 민다. 페달('a'/'b' 토글)은 어느 프레임이
    "개입"인지 label 만 표시하고, 그 프레임을 learner 가 가중 학습한다.

환경변수 override:
  RESIDUAL_INTERVENTION_WORKDIR      : actor/learner 파일 통신 폴더
  RESIDUAL_INTERVENTION_SLOW_CKPT    : frozen slow(base) 정책 ckpt (teleop 판과 동일 base 가능)
  RESIDUAL_INTERVENTION_CONFIG_NAME  : diffusion_policy/config/residual_policy 하위 config 이름
  INTERVENTION_SAMPLE_WEIGHT         : 개입 프레임 가중치(>1)
  RESIDUAL_INTERVENTION_LR / _EPOCHS_PER_ROUND / _FIRST_EPOCHS / _BATCH_SIZE /
    _MIN_EPISODES / _MAX_SAMPLES / _NUM_WORKERS / _DEVICE
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# frozen slow(base) 정책 — hand 16D (pose9+hand7), wrench_encoder 계열. teleop 판과 동일 base.
SLOW_CKPT = os.environ.get(
    "RESIDUAL_INTERVENTION_SLOW_CKPT",
    "/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt",
)

# residual 정책/데이터셋을 정의하는 hydra config (config_path=diffusion_policy/config)
# 개입 전용 task(hand_intervention: dataset 에 intervention_key 존재)를 바인딩.
RESIDUAL_CONFIG_NAME = os.environ.get(
    "RESIDUAL_INTERVENTION_CONFIG_NAME", "residual_policy/hand_intervention_mlp")

# actor/learner 파일 통신 폴더 (같은 머신 또는 공유 FS). teleop 판과 반드시 분리.
ONLINE_WORKDIR = os.environ.get(
    "RESIDUAL_INTERVENTION_WORKDIR",
    os.path.join(ROOT, "data/online_runs/run_hand_intervention"))
ACTOR_OUTPUT_DIR = os.path.join(ONLINE_WORKDIR, "actor_episodes")

# ── 학습 하이퍼파라미터 (intervention learner) ─────────────────────────────────
LR = float(os.environ.get("RESIDUAL_INTERVENTION_LR", 1.0e-4))
EPOCHS_PER_ROUND = int(os.environ.get("RESIDUAL_INTERVENTION_EPOCHS_PER_ROUND", 40))
FIRST_EPOCHS = int(os.environ.get("RESIDUAL_INTERVENTION_FIRST_EPOCHS", 120))
BATCH_SIZE = int(os.environ.get("RESIDUAL_INTERVENTION_BATCH_SIZE", 64))
NUM_WORKERS = int(os.environ.get("RESIDUAL_INTERVENTION_NUM_WORKERS", 0))
MIN_EPISODES_BEFORE_TRAIN = int(os.environ.get("RESIDUAL_INTERVENTION_MIN_EPISODES", 2))
# ── 대배치 DAgger 게이팅 (원본 CR-DAgger num_episodes_before_first_training / update-every-N) ──
FIRST_TRAIN_EPISODES = int(os.environ.get("RESIDUAL_INTERVENTION_FIRST_TRAIN_EPISODES", 50))
UPDATE_EVERY_N_EPISODES = int(os.environ.get("RESIDUAL_INTERVENTION_UPDATE_EVERY_N", 10))
# epoch 당 샘플 수 상한(0=len(dataset) 만큼 한 pass). 가중 샘플러의 num_samples 로 쓰인다.
MAX_SAMPLES_PER_EPOCH = int(os.environ.get("RESIDUAL_INTERVENTION_MAX_SAMPLES_PER_EPOCH", 0))
DEVICE = os.environ.get("RESIDUAL_INTERVENTION_DEVICE", "cuda:0")

# 개입(is_intervention=1) 프레임을 학습에서 몇 배 가중할지. cr-dagger DynamicDataset 대응.
# 안 민 프레임(residual≈0 negative)도 함께 학습해 nominal 상태 과도교정을 막되,
# 실제 교정 신호가 있는 개입 프레임을 이 배수만큼 자주 뽑는다.
INTERVENTION_SAMPLE_WEIGHT = float(os.environ.get("INTERVENTION_SAMPLE_WEIGHT", 5.0))

# 개입 '시작 구간' 집중 샘플링(원본 CR-DAgger dense-after). 각 개입 onset 이후 이 스텝 수만
# 가중해 뽑는다. 논문: "dense after start"(100%) > around(75) > uniform(70); 시작 '직전'
# (실패 징후=negative) 은 가중 안 함. 0 이면 개입 전체 균등 가중(구버전 동작).
CORRECTION_START_HORIZON = int(os.environ.get("INTERVENTION_CORRECTION_START_HORIZON", 8))

# residual head EMA (작아서 이득 적지만 옵션 유지)
USE_EMA = os.environ.get("RESIDUAL_INTERVENTION_USE_EMA", "1") == "1"

# 학습 타깃에서 임피던스 추종지연을 걷어낼지(online_learning/lag_model.py). 끄면 raw residual.
REMOVE_LAG = os.environ.get("RESIDUAL_INTERVENTION_REMOVE_LAG", "1") == "1"
# 개입하지 않은 프레임의 타깃을 정확히 0 으로 강제할지. CR-DAgger 전제("개입 안 함 = base 가
# 옳음")를 그대로 라벨에 반영한다. 끄면 지연 제거 후 잔차(노이즈)를 head 가 맞추려 들어
# nominal 구간에서도 계속 명령을 낸다.
ZERO_NOMINAL = os.environ.get("RESIDUAL_INTERVENTION_ZERO_NOMINAL", "1") == "1"

SEND_TRANSITIONS = True
