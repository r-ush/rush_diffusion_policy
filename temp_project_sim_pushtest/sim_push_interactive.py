#!/usr/bin/env python
"""
[TEMP PROJECT] 실시간 inference + 화살표 push 상호작용 시뮬레이션
==============================================================

목적:
  학습된 diffusion policy로 실시간 추론을 돌리면서, 사용자가 키보드 화살표로
  "사람이 로봇을 미는 것"을 흉내낼 수 있는 인터랙티브 시뮬레이터.
  CR-DAgger식 human-correction 루프의 '느낌'을 실제 로봇 없이 눈으로 확인하기 위함.

조작:
  ↑ (Up)    : x+ 방향으로 미는 것처럼 (앞으로)
  ↓ (Down)  : x- 방향
  ← (Left)  : y- 방향
  → (Right) : y+ 방향
  W / S     : z+ / z- (높이)
  (누르고 있으면 계속 밀림)
  R         : 로봇 pose를 demo 시작 위치로 리셋
  SPACE     : push 궤적 trace 지우기
  ESC / Q   : 종료

╔══════════════════════════════════════════════════════════════════════════╗
║ ⚠️  중요한 한계 (반드시 이해하고 볼 것)                                      ║
║  이 policy는 이미지 조건부라서, 시뮬레이션 EE 상태에 맞는 렌더링 화면이       ║
║  없다. 그래서 여기서는 '학습 데이터셋 demo의 카메라 이미지를 재생'해서        ║
║  policy에 입력으로 넣는다. 즉 policy는 demo 궤적을 따라가려 하고, 사용자      ║
║  push는 그 위에서 EE 실제 위치만 밀어낸다.                                   ║
║  → policy는 push를 '보지' 못한다(이미지가 고정 재생이므로). 따라서 이것은     ║
║    "상호작용 메커니즘 / correction 루프의 시각적 데모"이지, 닫힌 루프 task    ║
║    성공을 재현하는 물리 시뮬레이션이 아니다.                                 ║
║  push한 만큼 achieved pose가 어긋나고, policy가 절대 target으로 다시          ║
║  끌어당기는 모습(=compliant impedance 하에서 사람이 밀었다 놓는 상황)을       ║
║  관찰하는 용도.                                                             ║
╚══════════════════════════════════════════════════════════════════════════╝

실행:
  conda activate robodiff
  cd /home/rush/rush_diffusion_policy
  python temp_project_sim_pushtest/sim_push_interactive.py \
      --ckpt data/outputs/logistic_box_unet_abs/checkpoints/epoch=0700-train_loss=0.001.ckpt \
      --dataset /home/rush/Desktop/Datasets/20260630_195919_diffusion_des.hdf5 \
      --demo 0
"""

import os
import sys
import time
import argparse
from collections import deque

import numpy as np
import torch
import dill
import hydra
import h5py
import pygame
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

# 프로젝트 루트를 path에 추가 (스크립트가 하위 폴더에 있으므로)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ----------------------------------------------------------------------------
# 회전 표현 변환
# ----------------------------------------------------------------------------
def rot6d_to_matrix(rot6d):
    """rot6d(6,) -> rotation matrix(3,3)"""
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=1)


def rot6d_to_quat(rot6d):
    return R.from_matrix(rot6d_to_matrix(rot6d)).as_quat()  # xyzw


# ----------------------------------------------------------------------------
# Policy 로드
# ----------------------------------------------------------------------------
def load_policy(ckpt_path, device):
    print(f"[INFO] 체크포인트 로드: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval().to(device)
    policy.num_inference_steps = 8  # 반응성 위해 낮춤 (기본보다 빠르게)
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    print(f"[INFO] policy 준비 완료 (horizon={policy.horizon}, "
          f"n_obs={policy.n_obs_steps}, n_action={policy.n_action_steps})")
    return policy, cfg


# ----------------------------------------------------------------------------
# 데이터셋에서 한 demo의 이미지/pose 시퀀스 로드 (이미지 조건부용)
# ----------------------------------------------------------------------------
def load_demo(dataset_path, demo_idx):
    with h5py.File(dataset_path, "r") as f:
        d = f[f"data/demo_{demo_idx}"]
        images = np.array(d["obs/image0"])       # (T, H, W, 3) uint8
        pose = np.array(d["obs/robot_pose_L"])   # (T, 3)
        quat = np.array(d["obs/robot_quat_L"])   # (T, 4)
    print(f"[INFO] demo_{demo_idx} 로드: images={images.shape}, pose={pose.shape}")
    return images, pose, quat


# ----------------------------------------------------------------------------
# 좌표 변환: world (x=forward, y=left/right) -> screen
# ----------------------------------------------------------------------------
class WorldView:
    def __init__(self, x_range, y_range, rect):
        # rect = (left, top, w, h) 픽셀 영역
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.left, self.top, self.w, self.h = rect

    def to_screen(self, wx, wy):
        # world x(forward) -> 화면 위쪽(작은 screen y)
        # world y          -> 화면 오른쪽(큰 screen x)
        fy = (wx - self.x_min) / (self.x_max - self.x_min)  # 0(bottom)..1(top)
        fx = (wy - self.y_min) / (self.y_max - self.y_min)  # 0(left)..1(right)
        sx = self.left + fx * self.w
        sy = self.top + (1.0 - fy) * self.h
        return int(sx), int(sy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="data/outputs/logistic_box_unet_abs/checkpoints/epoch=0700-train_loss=0.001.ckpt")
    parser.add_argument("--dataset", default="/home/rush/Desktop/Datasets/20260630_195919_diffusion_des.hdf5")
    parser.add_argument("--demo", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--follow_gain", type=float, default=0.35,
                        help="policy target으로 EE가 끌려가는 정도 (impedance 추종)")
    parser.add_argument("--push_gain", type=float, default=0.008,
                        help="화살표 한 tick당 push 변위 [m]")
    parser.add_argument("--img_advance", type=int, default=1,
                        help="policy tick당 재생 이미지 인덱스 증가량")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[WARN] CUDA 사용 불가 -> CPU (추론이 느릴 수 있음)")

    policy, cfg = load_policy(args.ckpt, device)
    shape_meta = cfg.task.shape_meta
    n_obs = cfg.n_obs_steps

    images, demo_pose, demo_quat = load_demo(args.dataset, args.demo)
    T_demo = len(images)
    img_h, img_w = images.shape[1], images.shape[2]

    # 시뮬레이션 상태: achieved(실제) EE pose. demo 시작 위치에서 출발.
    init_pose = demo_pose[0].astype(np.float64).copy()
    achieved_pos = init_pose.copy()
    achieved_quat = demo_quat[0].astype(np.float64).copy()

    # obs history (최근 n_obs step)
    img_idx = 0
    img_hist = deque([images[0]] * n_obs, maxlen=n_obs)
    pos_hist = deque([achieved_pos.copy()] * n_obs, maxlen=n_obs)
    quat_hist = deque([achieved_quat.copy()] * n_obs, maxlen=n_obs)

    # view 범위 (demo pose 기준 여유있게)
    xr = (demo_pose[:, 0].min() - 0.08, demo_pose[:, 0].max() + 0.08)
    yr = (demo_pose[:, 1].min() - 0.10, demo_pose[:, 1].max() + 0.10)

    # ---------------- pygame 셋업 ----------------
    pygame.init()
    IMG_SCALE = 2
    img_disp_w, img_disp_h = img_w * IMG_SCALE, img_h * IMG_SCALE  # 640x480
    plot_w, plot_h = 480, img_disp_h
    pad = 12
    text_h = 120
    win_w = pad + img_disp_w + pad + plot_w + pad
    win_h = pad + img_disp_h + pad + text_h
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("[TEMP] Diffusion Policy - Interactive Push Sim")
    font = pygame.font.SysFont("monospace", 15)
    font_big = pygame.font.SysFont("monospace", 18, bold=True)
    clock = pygame.time.Clock()

    view = WorldView(xr, yr, (pad + img_disp_w + pad, pad, plot_w, plot_h))

    trace = deque(maxlen=400)   # achieved EE 궤적
    target_trace = deque(maxlen=400)

    print("[INFO] 시작! 창을 클릭해 포커스를 준 뒤 화살표로 밀어보세요.")

    running = True
    target_pos = achieved_pos.copy()
    last_infer_dt = 0.0

    while running:
        # ---------------- 이벤트 처리 ----------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    achieved_pos = init_pose.copy()
                    achieved_quat = demo_quat[0].astype(np.float64).copy()
                    img_idx = 0
                    trace.clear(); target_trace.clear()
                    print("[INFO] 리셋")
                elif event.key == pygame.K_SPACE:
                    trace.clear(); target_trace.clear()

        # ---------------- 눌린 키 -> push 벡터 ----------------
        keys = pygame.key.get_pressed()
        push = np.zeros(3)
        if keys[pygame.K_UP]:    push[0] += args.push_gain   # x+
        if keys[pygame.K_DOWN]:  push[0] -= args.push_gain   # x-
        if keys[pygame.K_LEFT]:  push[1] -= args.push_gain   # y-
        if keys[pygame.K_RIGHT]: push[1] += args.push_gain   # y+
        if keys[pygame.K_w]:     push[2] += args.push_gain   # z+
        if keys[pygame.K_s]:     push[2] -= args.push_gain   # z-
        pushing = np.linalg.norm(push) > 1e-9

        # ---------------- obs 구성 ----------------
        env_obs = {
            "image0": np.stack(list(img_hist)),        # (n_obs, H, W, 3) uint8
            "robot_pose_L": np.stack(list(pos_hist)),  # (n_obs, 3)
            "robot_quat_L": np.stack(list(quat_hist)), # (n_obs, 4)
        }

        # ---------------- policy 추론 ----------------
        t0 = time.time()
        with torch.no_grad():
            obs_dict_np = get_real_obs_dict(env_obs=env_obs, shape_meta=shape_meta)
            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            action = result["action"][0].detach().cpu().numpy()  # (n_action, 9)
        last_infer_dt = time.time() - t0

        # 첫 action step을 목표 pose로 사용 (절대 표현)
        target_pos = action[0, :3].astype(np.float64)
        target_quat = rot6d_to_quat(action[0, 3:9])

        # ---------------- impedance 추종 + human push ----------------
        # 실제 EE는 policy target으로 끌려가되, push만큼 밀려난다.
        achieved_pos = achieved_pos + args.follow_gain * (target_pos - achieved_pos) + push
        # 방향은 policy target을 따라가게 둠 (push는 위치만 흉내)
        achieved_quat = target_quat

        # ---------------- 이미지 재생 인덱스 전진 ----------------
        img_idx = (img_idx + args.img_advance) % T_demo

        # ---------------- history 갱신 ----------------
        img_hist.append(images[img_idx])
        pos_hist.append(achieved_pos.copy())
        quat_hist.append(achieved_quat.copy())

        trace.append(achieved_pos.copy())
        target_trace.append(target_pos.copy())

        # ==================== 렌더링 ====================
        screen.fill((25, 25, 30))

        # (1) 카메라 이미지 (재생중인 demo 이미지)
        cur_img = images[img_idx]  # (H,W,3) RGB uint8
        surf = pygame.surfarray.make_surface(np.transpose(cur_img, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (img_disp_w, img_disp_h))
        screen.blit(surf, (pad, pad))
        screen.blit(font.render("policy 입력 이미지 (demo 재생)", True, (200, 200, 200)),
                    (pad + 4, pad + 4))

        # (2) top-down x-y plot
        px, py, pw, ph = view.left, view.top, view.w, view.h
        pygame.draw.rect(screen, (40, 40, 50), (px, py, pw, ph))
        pygame.draw.rect(screen, (90, 90, 110), (px, py, pw, ph), 2)

        # 축 라벨
        screen.blit(font.render("x+ (forward)", True, (140, 140, 160)), (px + pw//2 - 40, py + 4))
        screen.blit(font.render("x-", True, (140, 140, 160)), (px + pw//2 - 8, py + ph - 20))
        screen.blit(font.render("y-", True, (140, 140, 160)), (px + 6, py + ph//2))
        screen.blit(font.render("y+", True, (140, 140, 160)), (px + pw - 24, py + ph//2))

        # target 궤적 (초록)
        if len(target_trace) > 1:
            pts = [view.to_screen(p[0], p[1]) for p in target_trace]
            pygame.draw.lines(screen, (60, 160, 90), False, pts, 1)
        # achieved 궤적 (파랑)
        if len(trace) > 1:
            pts = [view.to_screen(p[0], p[1]) for p in trace]
            pygame.draw.lines(screen, (80, 140, 230), False, pts, 2)

        # target 점 (초록)
        tsx, tsy = view.to_screen(target_pos[0], target_pos[1])
        pygame.draw.circle(screen, (80, 220, 120), (tsx, tsy), 6, 2)
        # achieved EE 점 (파랑, 채움)
        asx, asy = view.to_screen(achieved_pos[0], achieved_pos[1])
        pygame.draw.circle(screen, (90, 160, 255), (asx, asy), 9)

        # push 화살표 (빨강)
        if pushing:
            end = achieved_pos[:2] + push[:2] / max(np.linalg.norm(push[:2]), 1e-9) * 0.05
            esx, esy = view.to_screen(end[0], end[1])
            pygame.draw.line(screen, (255, 80, 80), (asx, asy), (esx, esy), 4)
            pygame.draw.circle(screen, (255, 80, 80), (esx, esy), 5)

        # (3) 텍스트 정보
        ty0 = pad + img_disp_h + pad
        err = np.linalg.norm(achieved_pos - target_pos)
        lines = [
            f"achieved  x={achieved_pos[0]:+.3f}  y={achieved_pos[1]:+.3f}  z={achieved_pos[2]:+.3f}",
            f"target    x={target_pos[0]:+.3f}  y={target_pos[1]:+.3f}  z={target_pos[2]:+.3f}   |err|={err*1000:5.1f} mm",
            f"infer {last_infer_dt*1000:5.1f} ms   img_idx={img_idx}/{T_demo}   PUSH={'ON  '+np.array2string(np.sign(push),precision=0) if pushing else 'off'}",
        ]
        for i, ln in enumerate(lines):
            col = (255, 120, 120) if (i == 2 and pushing) else (210, 210, 210)
            screen.blit(font.render(ln, True, col), (pad, ty0 + i * 20))
        help_txt = "↑x+ ↓x- ←y- →y+  W/S:z  |  R:reset  SPACE:clear trace  ESC/Q:quit"
        screen.blit(font_big.render(help_txt, True, (170, 200, 170)),
                    (pad, ty0 + 3 * 20 + 4))

        pygame.display.flip()
        clock.tick(30)  # 최대 30fps (추론이 느리면 자동으로 그보다 낮아짐)

    pygame.quit()
    print("[INFO] 종료")


if __name__ == "__main__":
    main()
