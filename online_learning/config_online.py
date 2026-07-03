"""
온라인 DAgger 학습 공유 설정 (actor & learner 둘 다 이 파일을 import).
CR-DAgger의 online_learning/configs/config_v1.py에 대응.
"""
import os

# ── 경로 ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# base policy 체크포인트 (actor가 로드해서 inference, learner가 fine-tune 시작점)
BASE_CKPT = os.path.join(
    ROOT, "data/outputs/logistic_box_unet_abs/checkpoints/epoch=0700-train_loss=0.001.ckpt")

# actor/learner가 파일로 통신하는 작업 폴더 (같은 머신 또는 공유 FS여야 함)
# 환경변수 ONLINE_WORKDIR 로 덮어쓸 수 있음 (GUI 데모가 임시 폴더를 주입할 때 사용)
ONLINE_WORKDIR = os.environ.get("ONLINE_WORKDIR", os.path.join(ROOT, "data/online_runs/run1"))

# actor가 원본 에피소드(replay_buffer.zarr)를 저장할 폴더 (rush_eval env가 사용)
ACTOR_OUTPUT_DIR = os.path.join(ONLINE_WORKDIR, "actor_episodes")

# forgetting 완화를 위해 base 학습 데이터 일부를 섞을지 (0이면 안 섞음, 온라인 데이터만)
BASE_DATASET_PATH = "/home/rush/Desktop/Datasets/20260630_195919_diffusion_des.hdf5"
NUM_BASE_DEMOS_TO_MIX = 0   # 예: 10 이면 base demo 0~9를 누적셋에 함께 넣음

# ── 학습 하이퍼파라미터 (learner) ────────────────────────────────────────────
# (환경변수로 덮어쓰기 가능 — GUI 데모가 가볍게 돌리려고 사용)
LR = float(os.environ.get("ONLINE_LR", 1.0e-5))                # fine-tune이므로 낮게
EPOCHS_PER_ROUND = int(os.environ.get("ONLINE_EPOCHS_PER_ROUND", 30))
BATCH_SIZE = int(os.environ.get("ONLINE_BATCH_SIZE", 16))
NUM_WORKERS = int(os.environ.get("ONLINE_NUM_WORKERS", 4))
MIN_EPISODES_BEFORE_TRAIN = int(os.environ.get("ONLINE_MIN_EPISODES", 2))
# 한 epoch당 학습 샘플 수 상한 (0=전체 사용). 에피소드가 길어도 라운드를 빠르게 끝내
# v1 가중치가 금방 나오게 하려는 용도. 초과분은 매 epoch 랜덤 subsample.
MAX_SAMPLES_PER_EPOCH = int(os.environ.get("ONLINE_MAX_SAMPLES_PER_EPOCH", 0))
DEVICE = os.environ.get("ONLINE_DEVICE", "cuda:0")

# ── actor 동작 ───────────────────────────────────────────────────────────────
# actor는 매 에피소드 시작 시 weights mailbox를 폴링해 새 버전이 있으면 hot-swap.
SEND_TRANSITIONS = True            # False면 순수 평가(데이터 안 보냄)
