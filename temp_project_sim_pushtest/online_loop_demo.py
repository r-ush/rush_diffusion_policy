#!/usr/bin/env python
"""
[TEMP PROJECT] 로봇 없이 온라인 학습 루프 전체를 GUI로 지켜보는 데모
==================================================================

한 화면에서:
  1) 화살표로 로봇을 밀어(사람 correction) 궤적을 만든다
  2) ENTER 로 그 구간을 correction 에피소드로 learner에 전송
  3) 백그라운드 learner가 fine-tune → 새 가중치 발행 (상태가 화면에 표시됨)
  4) GUI가 새 가중치를 hot-swap → policy가 바뀜 (버전 표시 + 화면 플래시)

즉 "사람이 밀기 → correction 데이터 → learner 학습 → 가중치 hot-swap → policy 변화"
전 과정을 로봇 없이 눈으로 확인한다.

╔══════════════════════════════════════════════════════════════════════════╗
║ ⚠️  한계 (sim_push_interactive.py와 동일)                                   ║
║  policy는 이미지 조건부인데 여기선 demo 이미지를 재생해 넣는다. 따라서       ║
║  policy는 push를 '보지' 못하고, 학습 후 policy 출력이 눈에 띄게 "민 쪽으로   ║
║  간다"는 보장은 없다(데이터도 적음). 이 데모의 목적은 **온라인 루프의        ║
║  데이터/학습/가중치 흐름을 시각적으로 확인**하는 것이지 task 학습 성공을      ║
║  보이는 게 아니다. 가중치 버전이 오르고 policy가 실제로 swap되는 것은 확실히   ║
║  보인다.                                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝

조작:
  ↑x+ ↓x- ←y- →y+  W/S:z    (누르고 있으면 계속 밀림 = correction)
  ENTER  : 지금까지 기록한 구간을 correction 에피소드로 learner에 전송
  BACKSPACE : 기록 버퍼 비우기(전송 안 함)
  R : 로봇 pose 리셋   SPACE : 화면 궤적 지우기   ESC/Q : 종료(learner도 종료)

실행:
  conda activate robodiff
  cd /home/rush/rush_diffusion_policy
  python temp_project_sim_pushtest/online_loop_demo.py
"""
import os
import sys
import time
import shutil
import subprocess
from collections import deque

# ── online 루프용 임시 작업폴더/설정을 config import 전에 주입 ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
WORKDIR = os.path.join(ROOT, "data/online_runs/gui_demo")
if os.path.exists(WORKDIR):
    shutil.rmtree(WORKDIR, ignore_errors=True)
os.makedirs(WORKDIR, exist_ok=True)
os.environ["ONLINE_WORKDIR"] = WORKDIR
os.environ.setdefault("ONLINE_EPOCHS_PER_ROUND", "12")  # 데모니까 짧게
os.environ.setdefault("ONLINE_MIN_EPISODES", "1")       # 1개만 받아도 학습
os.environ.setdefault("ONLINE_NUM_WORKERS", "0")
os.environ.setdefault("ONLINE_BATCH_SIZE", "8")
# 에피소드가 길어도 한 라운드가 빨리 끝나 v1이 금방 나오게 (epoch당 샘플 상한)
os.environ.setdefault("ONLINE_MAX_SAMPLES_PER_EPOCH", "128")

# GUI 기록 버퍼 최대 길이 (이보다 길게 밀면 오래된 것부터 버림 → 에피소드 비대화 방지)
MAX_REC = 250

import numpy as np
import torch
import dill
import hydra
import h5py
import pygame
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from online_learning import config_online as C
from online_learning.mailbox import FileMailbox
from online_learning.relabel_utils import relabel_episode_to_hdf5

OmegaConf.register_new_resolver("eval", eval, replace=True)


def rot6d_to_quat(rot6d):
    a1, a2 = rot6d[:3], rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return R.from_matrix(np.stack((b1, b2, b3), axis=1)).as_quat()


class WorldView:
    def __init__(self, x_range, y_range, rect):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.left, self.top, self.w, self.h = rect

    def to_screen(self, wx, wy):
        fy = (wx - self.x_min) / (self.x_max - self.x_min)
        fx = (wy - self.y_min) / (self.y_max - self.y_min)
        return (int(self.left + fx * self.w),
                int(self.top + (1.0 - fy) * self.h))


def main():
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("ONLINE_DEVICE", str(device))

    # ── 백그라운드 learner 프로세스 실행 ──
    learner_log = open(os.path.join(WORKDIR, "learner.log"), "w")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    print("[demo] learner 프로세스 시작 (로딩에 10~20초 걸릴 수 있음)...")
    learner_proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "online_learning/finetune_teleop_learner.py")],
        env=env, stdout=learner_log, stderr=subprocess.STDOUT)

    mailbox = FileMailbox(WORKDIR)

    # ── GUI 쪽 base policy 로드 (inference용) ──
    print(f"[demo] base 체크포인트 로드: {C.BASE_CKPT}")
    payload = torch.load(open(C.BASE_CKPT, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(device)
    policy.num_inference_steps = 8
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    shape_meta = cfg.task.shape_meta
    n_obs = cfg.n_obs_steps
    policy_version = -1

    # ── demo 이미지/시작 pose 로드 (이미지 조건부용 재생 소스) ──
    with h5py.File(C.BASE_DATASET_PATH, "r") as f:
        d = f["data/demo_0"]
        images = np.array(d["obs/image0"]).astype(np.uint8)
        demo_pose = np.array(d["obs/robot_pose_L"])
        demo_quat = np.array(d["obs/robot_quat_L"])
    T_demo = len(images)
    img_h, img_w = images.shape[1], images.shape[2]

    init_pose = demo_pose[0].astype(np.float64).copy()
    achieved_pos = init_pose.copy()
    achieved_quat = demo_quat[0].astype(np.float64).copy()
    img_idx = 0
    img_hist = deque([images[0]] * n_obs, maxlen=n_obs)
    pos_hist = deque([achieved_pos.copy()] * n_obs, maxlen=n_obs)
    quat_hist = deque([achieved_quat.copy()] * n_obs, maxlen=n_obs)

    # correction 기록 버퍼 (전송 대상) — 최근 MAX_REC 스텝만 유지
    rec_img = deque(maxlen=MAX_REC)
    rec_pose = deque(maxlen=MAX_REC)
    rec_quat = deque(maxlen=MAX_REC)
    rec_corr = deque(maxlen=MAX_REC)
    episodes_sent = 0
    swap_flash = 0.0

    xr = (demo_pose[:, 0].min() - 0.08, demo_pose[:, 0].max() + 0.08)
    yr = (demo_pose[:, 1].min() - 0.10, demo_pose[:, 1].max() + 0.10)

    follow_gain, push_gain = 0.35, 0.008

    # ── pygame ──
    pygame.init()
    SCALE = 2
    iw, ih = img_w * SCALE, img_h * SCALE
    plot_w = 460
    pad = 12
    panel_h = 175
    win_w = pad + iw + pad + plot_w + pad
    win_h = pad + ih + pad + panel_h
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("[TEMP] Online DAgger Loop - No-Robot GUI Demo")
    font = pygame.font.SysFont("monospace", 14)
    fontb = pygame.font.SysFont("monospace", 17, bold=True)
    clock = pygame.time.Clock()
    view = WorldView(xr, yr, (pad + iw + pad, pad, plot_w, ih))
    trace, target_trace = deque(maxlen=500), deque(maxlen=500)

    def send_episode():
        nonlocal episodes_sent
        if len(rec_pose) < 25:
            print(f"[demo] 기록이 짧아 전송 취소 ({len(rec_pose)} < 25). 더 움직여주세요.")
            return
        out = os.path.join(WORKDIR, "gui_episode.hdf5")
        relabel_episode_to_hdf5(np.stack(list(rec_img)), np.stack(list(rec_pose)),
                                np.stack(list(rec_quat)), out)
        mailbox.send_episode(out)
        episodes_sent += 1
        n_corr = int(np.sum(list(rec_corr)))
        print(f"[demo] 에피소드 전송 #{episodes_sent} (len={len(rec_pose)}, "
              f"correction step={n_corr})")
        rec_img.clear(); rec_pose.clear(); rec_quat.clear(); rec_corr.clear()

    print("[demo] 시작! 창을 클릭해 포커스를 준 뒤 화살표로 밀어보세요.")
    running = True
    target_pos = achieved_pos.copy()
    infer_ms = 0.0

    while running:
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
                elif event.key == pygame.K_SPACE:
                    trace.clear(); target_trace.clear()
                elif event.key == pygame.K_RETURN:
                    send_episode()
                elif event.key == pygame.K_BACKSPACE:
                    rec_img.clear(); rec_pose.clear(); rec_quat.clear(); rec_corr.clear()
                    print("[demo] 기록 버퍼 비움")

        keys = pygame.key.get_pressed()
        push = np.zeros(3)
        if keys[pygame.K_UP]:    push[0] += push_gain
        if keys[pygame.K_DOWN]:  push[0] -= push_gain
        if keys[pygame.K_LEFT]:  push[1] -= push_gain
        if keys[pygame.K_RIGHT]: push[1] += push_gain
        if keys[pygame.K_w]:     push[2] += push_gain
        if keys[pygame.K_s]:     push[2] -= push_gain
        pushing = np.linalg.norm(push) > 1e-9

        # ── 추론 ──
        env_obs = {"image0": np.stack(list(img_hist)),
                   "robot_pose_L": np.stack(list(pos_hist)),
                   "robot_quat_L": np.stack(list(quat_hist))}
        t0 = time.time()
        with torch.no_grad():
            od = get_real_obs_dict(env_obs=env_obs, shape_meta=shape_meta)
            od = dict_apply(od, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            action = policy.predict_action(od)["action"][0].detach().cpu().numpy()
        infer_ms = (time.time() - t0) * 1000
        target_pos = action[0, :3].astype(np.float64)
        target_quat = rot6d_to_quat(action[0, 3:9])

        # ── impedance 추종 + push ──
        achieved_pos = achieved_pos + follow_gain * (target_pos - achieved_pos) + push
        achieved_quat = target_quat
        img_idx = (img_idx + 1) % T_demo

        # 기록 (전송용) — 현재 화면에 재생중인 이미지 + 실제 pose + correction flag
        rec_img.append(images[img_idx].copy())
        rec_pose.append(achieved_pos.copy())
        rec_quat.append(achieved_quat.copy())
        rec_corr.append(1 if pushing else 0)

        img_hist.append(images[img_idx])
        pos_hist.append(achieved_pos.copy())
        quat_hist.append(achieved_quat.copy())
        trace.append(achieved_pos.copy())
        target_trace.append(target_pos.copy())

        # ── learner가 새 가중치 냈으면 hot-swap ──
        latest = mailbox.get_latest_weight_version()
        if latest is not None and latest > policy_version:
            wp = mailbox.load_weights(latest, map_location=device)
            if wp is not None:
                policy.load_state_dict(wp["state_dict"], strict=False)
                policy.eval().to(device)
                print(f"[demo] policy hot-swap: v{policy_version} -> v{latest}")
                policy_version = latest
                swap_flash = time.time()

        status = mailbox.read_status() or {}

        # ==================== 렌더링 ====================
        screen.fill((22, 22, 27))
        # 카메라 이미지
        surf = pygame.surfarray.make_surface(np.transpose(images[img_idx], (1, 0, 2)))
        surf = pygame.transform.scale(surf, (iw, ih))
        screen.blit(surf, (pad, pad))
        screen.blit(font.render("policy 입력 이미지 (demo 재생)", True, (200, 200, 200)),
                    (pad + 4, pad + 4))

        # xy plot
        px, py, pw, ph = view.left, view.top, view.w, view.h
        pygame.draw.rect(screen, (38, 38, 48), (px, py, pw, ph))
        pygame.draw.rect(screen, (90, 90, 110), (px, py, pw, ph), 2)
        screen.blit(font.render("x+", True, (130, 130, 150)), (px + pw//2 - 8, py + 3))
        screen.blit(font.render("y-", True, (130, 130, 150)), (px + 5, py + ph//2))
        screen.blit(font.render("y+", True, (130, 130, 150)), (px + pw - 22, py + ph//2))
        if len(target_trace) > 1:
            pygame.draw.lines(screen, (60, 160, 90), False,
                              [view.to_screen(p[0], p[1]) for p in target_trace], 1)
        if len(trace) > 1:
            pygame.draw.lines(screen, (80, 140, 230), False,
                              [view.to_screen(p[0], p[1]) for p in trace], 2)
        tsx, tsy = view.to_screen(target_pos[0], target_pos[1])
        pygame.draw.circle(screen, (80, 220, 120), (tsx, tsy), 6, 2)
        asx, asy = view.to_screen(achieved_pos[0], achieved_pos[1])
        pygame.draw.circle(screen, (90, 160, 255), (asx, asy), 9)
        if pushing:
            e = achieved_pos[:2] + push[:2] / max(np.linalg.norm(push[:2]), 1e-9) * 0.05
            esx, esy = view.to_screen(e[0], e[1])
            pygame.draw.line(screen, (255, 80, 80), (asx, asy), (esx, esy), 4)

        # hot-swap 플래시
        if time.time() - swap_flash < 1.2:
            screen.blit(fontb.render(f"⚡ policy SWAPPED -> v{policy_version}", True,
                        (255, 220, 90)), (px + 10, py + ph - 28))

        # ── 상태 패널 ──
        ty = pad + ih + pad
        col_actor = (150, 200, 255)
        col_learn = (255, 200, 130)
        lstate = status.get("state", "starting...")
        lstate_disp = {"idle": "대기(에피소드 기다림)",
                       "received_episode": "에피소드 수신",
                       "training": "학습중",
                       "published": "가중치 발행완료"}.get(lstate, lstate)
        lines = [
            (col_actor, f"[ACTOR/GUI] policy=v{policy_version}   에피소드 전송={episodes_sent}   "
                        f"기록버퍼={len(rec_pose)}   infer={infer_ms:4.0f}ms"),
            (col_actor, f"           achieved x={achieved_pos[0]:+.3f} y={achieved_pos[1]:+.3f} "
                        f"z={achieved_pos[2]:+.3f}   PUSH={'ON' if pushing else 'off'}"),
            (col_learn, f"[LEARNER]  상태={lstate_disp}   누적demo={status.get('num_demos','?')}   "
                        f"최신가중치=v{status.get('version','?')}"),
        ]
        if lstate == "training":
            lines.append((col_learn,
                f"           epoch {status.get('epoch','?')}/{status.get('num_epochs','?')}  "
                f"loss={status.get('last_loss', float('nan')):.5f}   "
                f"← 학습중, 지금 종료하지 마세요"))
        elif lstate == "published":
            lines.append((col_learn,
                f"           마지막 loss={status.get('last_loss', float('nan')):.5f}  "
                f"라운드시간={status.get('round_time_s','?')}s"))
        for i, (c, ln) in enumerate(lines):
            screen.blit(font.render(ln, True, c), (pad, ty + i * 20))

        help_txt = "↑x+ ↓x- ←y- →y+ W/S:z | ENTER:correction 전송  BACKSPACE:버퍼비움  R:리셋  ESC:종료"
        screen.blit(fontb.render(help_txt, True, (170, 200, 170)), (pad, ty + panel_h - 30))

        pygame.display.flip()
        clock.tick(30)

    # ── 정리 ──
    print("[demo] 종료 중... learner 프로세스 종료")
    learner_proc.terminate()
    try:
        learner_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        learner_proc.kill()
    learner_log.close()
    pygame.quit()
    print(f"[demo] 끝. (learner 로그: {os.path.join(WORKDIR, 'learner.log')})")


if __name__ == "__main__":
    main()
