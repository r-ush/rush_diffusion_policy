"""
온라인 residual DAgger 공유 설정 (actor & residual learner 둘 다 import).

full-finetune 경로의 config_online.py 대응. 차이:
  * base 를 fine-tune 하지 않는다. frozen slow base + 작은 residual head 만 학습.
  * 그래서 base ckpt 대신 **residual 설정 이름(config_name)** 과 **slow_ckpt** 를 지정.
  * 발행 가중치는 full state_dict 가 아니라 head(+선택 force_encoder) 만.

환경변수 override:
  RESIDUAL_ONLINE_WORKDIR  : actor/learner 파일 통신 폴더
  RESIDUAL_SLOW_CKPT       : frozen slow(base) 정책 ckpt (wrench_encoder 계열이어야 함)
  RESIDUAL_CONFIG_NAME     : diffusion_policy/config/residual_policy 하위 config 이름
  RESIDUAL_LR / _EPOCHS_PER_ROUND / _FIRST_EPOCHS / _BATCH_SIZE / _MIN_EPISODES / _MAX_SAMPLES
  RESIDUAL_DEVICE
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# frozen slow(base) 정책 — hand 16D (pose9+hand7), wrench_encoder 계열. SSD 경로.
SLOW_CKPT = os.environ.get(
    "RESIDUAL_SLOW_CKPT",
    "/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260714_insert_box_hand_rel/epoch=0500-train_loss=0.002.ckpt",
)

# residual 정책/데이터셋을 정의하는 hydra config (config_path=diffusion_policy/config)
RESIDUAL_CONFIG_NAME = os.environ.get("RESIDUAL_CONFIG_NAME", "residual_policy/hand_online_mlp")

# actor/learner 파일 통신 폴더 (같은 머신 또는 공유 FS)
ONLINE_WORKDIR = os.environ.get("RESIDUAL_ONLINE_WORKDIR", os.path.join(ROOT, "data/online_runs/run_hand_residual"))
ACTOR_OUTPUT_DIR = os.path.join(ONLINE_WORKDIR, "actor_episodes")

# ── 학습 하이퍼파라미터 (residual learner) ──────────────────────────────────
# head 가 tiny 라 라운드가 매우 빠름(<1s~수초). frozen base 라 망각 없음.
LR = float(os.environ.get("RESIDUAL_LR", 1.0e-4))
EPOCHS_PER_ROUND = int(os.environ.get("RESIDUAL_EPOCHS_PER_ROUND", 40))
# 첫 라운드는 부트스트랩용으로 더 많이 돈다(cr-dagger first_time_num_epochs).
FIRST_EPOCHS = int(os.environ.get("RESIDUAL_FIRST_EPOCHS", 120))
BATCH_SIZE = int(os.environ.get("RESIDUAL_BATCH_SIZE", 64))
NUM_WORKERS = int(os.environ.get("RESIDUAL_NUM_WORKERS", 0))
MIN_EPISODES_BEFORE_TRAIN = int(os.environ.get("RESIDUAL_MIN_EPISODES", 2))
# epoch 당 샘플 상한(0=전체). 교정 데이터에 correction 가중이 없으므로 단순 상한.
MAX_SAMPLES_PER_EPOCH = int(os.environ.get("RESIDUAL_MAX_SAMPLES_PER_EPOCH", 0))
DEVICE = os.environ.get("RESIDUAL_DEVICE", "cuda:0")

# residual head 는 EMA 를 쓸지 (residual head 는 작아서 EMA 이득이 적지만 옵션 유지)
USE_EMA = os.environ.get("RESIDUAL_USE_EMA", "1") == "1"

SEND_TRANSITIONS = True
