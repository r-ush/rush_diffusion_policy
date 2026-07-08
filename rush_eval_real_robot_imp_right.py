#!/home/vision/miniconda3/envs/robodiff/bin/python

# 실행코드 (오른팔 only inference)
# python rush_eval_real_robot_imp_right.py --input data/outputs/logistics_abs_10/epoch=0800-train_loss=0.000.ckpt --output data/results --steps_per_inference 12 --frequency 10 --num_inference_steps 12

"""
Usage:
(robodiff)$ python rush_eval_real_robot_imp_right.py -i <ckpt_path> -o <save_dir>

학습 체크포인트 정보 (logistics_abs_10):
  - obs: image0 (3x240x320), robot_pose_L (3,), robot_quat_L (4,)
    * obs key 이름은 오른팔 데이터라도 shape_meta 라벨(_L)을 그대로 사용
  - action: shape (9,) = pos(3) + rot6d(6), abs 표현
  - 오른팔만 inference

Recording control:
  Click the opencv window (make sure it's in focus).
  Press "C" to toggle correction mode ON/OFF. Turn it ON right before you
    physically push/guide the arm to correct it, and OFF once you let go.
    This flag is saved per control-step in the episode's `stage` field
    (0=policy, 1=human correction), used later by
    data_process/rush_replay_buffer_to_correction_hdf5.py to relabel data.
  Press "S" to stop the episode and KEEP the recorded data.
  Press "D" to stop the episode and DISCARD the recorded data.
  Press "Q" or Ctrl+C to exit program.
"""

# %%
# ============================================================
# huggingface_hub 버전 충돌 문제 해결 (local v1.x vs conda v0.11.1)
# 1. 임시로 .local 경로를 sys.path에서 제외
# 2. huggingface_hub를 conda 환경에서 먼저 임포트
# 3. 다시 sys.path 복구하여 torch(최신 쿠다지원) 등 다른 패키지는 정상 로드되게 함
# ============================================================
import sys
local_paths = [p for p in sys.path if '.local' in p]
sys.path = [p for p in sys.path if '.local' not in p]

import huggingface_hub

# 다시 원래대로 복구 (우선순위 유지)
sys.path = local_paths + sys.path

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from diffusion_policy.real_world.rush_real_env_rightarm_imp import RightarmRealEnvImp
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict,
    get_real_relative_obs_dict,
    get_abs_action_from_relative)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    """rotation 6d representation -> rotation vector"""
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R_mat = np.stack((b1, b2, b3), axis=1)
    return R.from_matrix(R_mat).as_rotvec()


def _plot_episode_force(timeseries_hdf5_dir, output_dir):
    """가장 최근 저장된 episode HDF5의 wrist_ft 힘(두 소스: aft + doosan)을
    fx/fy/fz 3개 subplot으로 그려 PNG로 저장한다.
    bae insert_plug eval의 force PNG 시각화를 rush(오른팔 임피던스)용으로 이식 +
    aft/doosan 2채널 비교로 확장."""
    import os
    import glob
    timeseries_hdf5_dir = pathlib.Path(timeseries_hdf5_dir)
    files = sorted(timeseries_hdf5_dir.glob('episode_*.hdf5'),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        print('[WARN] No timeseries HDF5 to plot.')
        return
    hdf5_path = files[-1]

    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        if 'wrist_ft' not in f:
            print('[WARN] wrist_ft group missing; skip force plot.')
            return
        g = f['wrist_ft']
        elapsed = (np.asarray(g['elapsed_s'], dtype=np.float64)
                   if 'elapsed_s' in g else np.empty((0,), dtype=np.float64))
        aft = (np.asarray(g['wrench_wrist_R'], dtype=np.float64)
               if 'wrench_wrist_R' in g else np.empty((0, 6), dtype=np.float64))
        doosan = (np.asarray(g['wrench_doosan_R'], dtype=np.float64)
                  if 'wrench_doosan_R' in g else np.empty((0, 6), dtype=np.float64))

    if len(elapsed) == 0 or (len(aft) == 0 and len(doosan) == 0):
        print('[WARN] No force samples to plot.')
        return
    t = elapsed - elapsed[0]

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    plot_dir = pathlib.Path(output_dir).joinpath('eval_debug')
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    labels = ('Fx', 'Fy', 'Fz')
    for i, (ax, label) in enumerate(zip(axes, labels)):
        if len(aft) > 0:
            n = min(len(t), len(aft))
            ax.plot(t[:n], aft[:n, i], color='#d62728', linewidth=1.6,
                    label='aft (/aft_sensor2/wrench)')
        if len(doosan) > 0:
            n = min(len(t), len(doosan))
            ax.plot(t[:n], doosan[:n, i], color='#1f77b4', linewidth=1.6,
                    label='doosan (get_tool_force)')
        ax.axhline(0.0, color='#333333', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f'{label} (N)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('elapsed time (s)')
    fig.suptitle(f'{hdf5_path.stem} wrist F/T force (aft vs doosan)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = plot_dir.joinpath(f'{hdf5_path.stem}_force_xyz.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f'Force plot saved: {png_path}')


def _safe_plot_episode_force(timeseries_hdf5_dir, output_dir):
    try:
        _plot_episode_force(timeseries_hdf5_dir, output_dir)
    except Exception as e:
        print(f'[WARNING] Failed to save force plot: {e}')


def _nearest_indices(query_time, sample_time):
    query_time = np.asarray(query_time, dtype=np.float64)
    sample_time = np.asarray(sample_time, dtype=np.float64)
    if len(sample_time) <= 1:
        return np.zeros((len(query_time),), dtype=np.int64)
    insert = np.searchsorted(sample_time, query_time, side='left')
    insert = np.clip(insert, 1, len(sample_time) - 1)
    left = insert - 1
    right = insert
    use_right = np.abs(sample_time[right] - query_time) < np.abs(query_time - sample_time[left])
    return np.where(use_right, right, left)


def _latest_episode_hdf5(timeseries_hdf5_dir):
    files = sorted(pathlib.Path(timeseries_hdf5_dir).glob('episode_*.hdf5'),
                   key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _plot_episode_action_3d_html(timeseries_hdf5_dir, output_dir, steps_per_inference):
    """가장 최근 episode HDF5로 정책 target action + 실제 궤적 + 힘 화살표를 담은
    인터랙티브 plotly 3D HTML을 저장한다. base(로봇) 프레임 기준.
    bae insert_plug eval의 3D HTML 시각화를 rush(오른팔, robot_pose_L / base frame)로 이식."""
    hdf5_path = _latest_episode_hdf5(timeseries_hdf5_dir)
    if hdf5_path is None:
        print('[WARN] No timeseries HDF5 to plot (3D HTML).')
        return

    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        if 'action_virtual_target' not in f or 'action' not in f['action_virtual_target']:
            print('[WARN] action_virtual_target/action missing; skip 3D HTML.')
            return
        ag = f['action_virtual_target']
        action = np.asarray(ag['action'], dtype=np.float64)
        a_elapsed = (np.asarray(ag['elapsed_s'], dtype=np.float64)
                     if 'elapsed_s' in ag else np.arange(len(action), dtype=np.float64))

        def _get(group, key, shape, dt=np.float64):
            if group in f and key in f[group]:
                return np.asarray(f[f'{group}/{key}'], dtype=dt)
            return np.empty(shape, dtype=dt)

        actual_elapsed = _get('actual', 'elapsed_s', (0,))
        actual_pose = _get('actual', 'robot_pose_L', (0, 3))
        actual_quat = _get('actual', 'robot_quat_L', (0, 4))
        ft_elapsed = _get('wrist_ft', 'elapsed_s', (0,))
        ft_wrench = _get('wrist_ft', 'wrench_wrist_R', (0, 6))

    if len(action) == 0:
        return
    n = len(action)
    spi = max(int(steps_per_inference), 1)
    action_pos = action[:, :3]
    inference_index = np.arange(n, dtype=np.int64) // spi
    m = min(len(a_elapsed), n)
    a_elapsed = a_elapsed[:m]
    action_pos = action_pos[:m]
    inference_index = inference_index[:m]

    import plotly.graph_objects as go
    fig = go.Figure()

    # 실제 로봇 궤적 (line)
    if len(actual_pose) > 0:
        fig.add_trace(go.Scatter3d(
            x=actual_pose[:, 0], y=actual_pose[:, 1], z=actual_pose[:, 2],
            mode='lines', name='real robot trajectory',
            line=dict(color='#111111', width=5),
            customdata=actual_elapsed,
            hovertemplate='real t=%{customdata:.3f}s<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>'))

    # 정책 target action (line+markers, inference index 로 색상)
    custom = np.column_stack([np.arange(m), inference_index, a_elapsed])
    fig.add_trace(go.Scatter3d(
        x=action_pos[:, 0], y=action_pos[:, 1], z=action_pos[:, 2],
        mode='lines+markers', name='policy target action',
        line=dict(color='#d62728', width=3),
        marker=dict(size=4, color=inference_index, colorscale='Turbo',
                    colorbar=dict(title='infer')),
        customdata=custom,
        hovertemplate=('cmd=%{customdata[0]:.0f}, infer=%{customdata[1]:.0f}<br>'
                       't=%{customdata[2]:.3f}s<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>')))

    # replan(새 inference) 경계 표시
    replan = np.flatnonzero(np.r_[True, np.diff(inference_index) != 0])
    if len(replan) > 0:
        fig.add_trace(go.Scatter3d(
            x=action_pos[replan, 0], y=action_pos[replan, 1], z=action_pos[replan, 2],
            mode='markers', name='new inference boundary',
            marker=dict(size=5, color='#ffffff', symbol='diamond',
                        line=dict(color='#111111', width=1.5)),
            hoverinfo='skip'))

    # 힘 화살표 (aft 손목 F/T 를 base 프레임으로 회전해서 TCP 위치에 표시)
    force_origin = np.empty((0, 3))
    if (len(actual_pose) > 0 and len(actual_quat) > 0
            and len(ft_wrench) > 0 and len(actual_elapsed) > 0):
        pose_idx = _nearest_indices(a_elapsed, actual_elapsed)
        ft_idx = _nearest_indices(a_elapsed, ft_elapsed)
        origin = actual_pose[pose_idx, :3]
        quat = actual_quat[pose_idx]
        rotmat = R.from_quat(quat).as_matrix()               # sensor/EE -> base
        fvec = np.einsum('nij,nj->ni', rotmat, ft_wrench[ft_idx, :3])
        good = (np.all(np.isfinite(origin), axis=1)
                & np.all(np.isfinite(fvec), axis=1)
                & (np.linalg.norm(fvec, axis=1) > 1e-6))
        origin = origin[good]
        fvec = fvec[good]
        if len(origin) > 0:
            max_arrows = 80
            if len(origin) > max_arrows:
                sel = np.unique(np.linspace(0, len(origin) - 1, max_arrows, dtype=np.int64))
                origin = origin[sel]
                fvec = fvec[sel]
            scale = np.nanpercentile(np.linalg.norm(fvec, axis=1), 95)
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            unit = fvec / scale
            end = origin + unit * 0.035
            lx, ly, lz = [], [], []
            for s, e in zip(origin, end):
                lx += [s[0], e[0], None]
                ly += [s[1], e[1], None]
                lz += [s[2], e[2], None]
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode='lines', name='force arrows (aft, base)',
                line=dict(color='#7e22ce', width=4), hoverinfo='skip'))
            fig.add_trace(go.Cone(
                x=end[:, 0], y=end[:, 1], z=end[:, 2],
                u=unit[:, 0], v=unit[:, 1], w=unit[:, 2],
                customdata=fvec, anchor='tip', sizemode='absolute', sizeref=0.012,
                colorscale=[[0.0, '#7e22ce'], [1.0, '#7e22ce']], showscale=False,
                name='force arrow heads',
                hovertemplate='Fx=%{customdata[0]:.2f}<br>Fy=%{customdata[1]:.2f}<br>Fz=%{customdata[2]:.2f} N<extra></extra>'))
            force_origin = origin

    # 축 범위(cube)
    centers = [action_pos]
    if len(actual_pose) > 0:
        centers.append(actual_pose[:, :3])
    if len(force_origin) > 0:
        centers.append(force_origin)
    centers = np.vstack(centers)
    c = centers.mean(axis=0)
    radius = max(float(np.ptp(centers, axis=0).max()) * 0.58, 0.01)

    fig.update_layout(
        title=f'{hdf5_path.stem} policy target actions (base frame)',
        template='plotly_white', width=1100, height=850,
        scene=dict(
            xaxis=dict(title='base x (m)', range=[c[0] - radius, c[0] + radius]),
            yaxis=dict(title='base y (m)', range=[c[1] - radius, c[1] + radius]),
            zaxis=dict(title='base z (m)', range=[c[2] - radius, c[2] + radius]),
            aspectmode='cube'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
        margin=dict(l=20, r=20, t=80, b=20))

    debug_dir = pathlib.Path(output_dir).joinpath('eval_debug')
    debug_dir.mkdir(parents=True, exist_ok=True)
    html_path = debug_dir.joinpath(f'{hdf5_path.stem}_action3d.html')
    fig.write_html(str(html_path), include_plotlyjs='cdn', full_html=True)
    print(f'Policy action 3D HTML saved: {html_path}')


def _safe_write_eval_diagnostics(timeseries_hdf5_dir, output_dir, steps_per_inference):
    """에피소드 종료 시 시각화 일괄 저장: 힘 그래프 PNG + 정책 action 3D HTML."""
    _safe_plot_episode_force(timeseries_hdf5_dir, output_dir)
    try:
        _plot_episode_action_3d_html(timeseries_hdf5_dir, output_dir, steps_per_inference)
    except Exception as e:
        print(f'[WARNING] Failed to save policy action 3D HTML: {e}')


@click.command()
@click.option('--input',  '-i', required=True,  help='Path to checkpoint (.ckpt)')
@click.option('--output', '-o', required=True,  help='Directory to save recording')
@click.option('--robot_ip', '-ri', default="192.168.111.50",
              help="Robot IP (not used by RightarmRealEnv but kept for compatibility)")
@click.option('--vis_camera_idx', default=0, type=int,
              help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=6, type=int,
              help="Number of action steps to execute per inference (action horizon).")
@click.option('--max_duration', '-md', default=60, type=float,
              help='Max duration for each episode in seconds.')
@click.option('--frequency', '-f', default=10, type=float,
              help="Control frequency in Hz.")
@click.option('--num_inference_steps', '-ni', default=16, type=int,
              help="DDIM denoising steps (more → slower but potentially better).")
@click.option('--record_wrench/--no_record_wrench', default=True,
              help="aft 손목 F/T(토픽)+관절각 timeseries HDF5 로깅 여부.")
@click.option('--wrench_topic', default='/aft_sensor2/wrench',
              help="오른팔 손목 F/T 센서 WrenchStamped 토픽 이름.")
@click.option('--record_doosan_force/--no_record_doosan_force', default=True,
              help="doosan api 외력(get_tool_force 서비스)도 함께 로깅.")
@click.option('--doosan_force_service', default='/dsr01/aux_control/get_tool_force',
              help="doosan api 외력 서비스 이름 (get_tool_force).")
@click.option('--plot_force/--no_plot_force', default=True,
              help="에피소드 종료 시 aft vs doosan 힘 그래프 PNG 저장.")
def main(input, output, robot_ip,
         vis_camera_idx, steps_per_inference, max_duration,
         frequency, num_inference_steps, record_wrench, wrench_topic,
         record_doosan_force, doosan_force_service, plot_force):

    # ===================== 1. Load checkpoint =====================
    ckpt_path = input
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # ===================== 2. Policy setup ========================
    action_offset = 0
    delta_action = False

    if 'diffusion' not in cfg.name:
        raise RuntimeError(f"Unsupported policy type: {cfg.name}")

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # DDIM inference steps
    policy.num_inference_steps = num_inference_steps
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    print(f"[INFO] Policy: {cfg.name}")
    print(f"[INFO] horizon={policy.horizon}, n_obs_steps={policy.n_obs_steps}, "
          f"n_action_steps={policy.n_action_steps}")

    # ===================== 2.5 Relative/abs pose repr =================
    # If not present in cfg, defaults to abs behavior.
    obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=None)
    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default=None)
    print(f"[DEBUG] cfg.task.pose_repr: {OmegaConf.select(cfg, 'task.pose_repr', default='NOT FOUND')}")
    print(f"[DEBUG] obs_pose_repr={obs_pose_repr}, action_pose_repr={action_pose_repr}")
    print(f"[DEBUG] action shape_meta: {cfg.task.shape_meta.action}")
    if obs_pose_repr is not None or action_pose_repr is not None:
        print(f"[INFO] pose_repr: obs={obs_pose_repr}, action={action_pose_repr}")

    # ===================== 3. Env setup ===========================
    dt = 1.0 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # (width, height) = (320, 240)
    n_obs_steps = cfg.n_obs_steps   # 2

    print(f"[INFO] obs_res: {obs_res}")
    print(f"[INFO] n_obs_steps: {n_obs_steps}")
    print(f"[INFO] steps_per_inference: {steps_per_inference}")
    if record_wrench or record_doosan_force:
        print(f"[INFO] Force/joint logging ON → {pathlib.Path(output).joinpath('timeseries_hdf5')}")
        if record_wrench:
            print(f"        · aft 손목 F/T  : {wrench_topic} → wrist_ft/wrench_wrist_R")
        if record_doosan_force:
            print(f"        · doosan api 외력: {doosan_force_service} → wrist_ft/wrench_doosan_R")
        if plot_force:
            print(f"        · 시각화(PNG+HTML): {pathlib.Path(output).joinpath('eval_debug')} "
                  f"(force 그래프 + 정책 action 3D, 에피소드 종료 시)")
    else:
        print("[INFO] Force/joint logging OFF (--record_wrench / --record_doosan_force 로 활성화)")

    with SharedMemoryManager() as shm_manager:
        with RightarmRealEnvImp(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,         # (320, 240)
            obs_float32=True,
            enable_multi_cam_vis=True,
            record_raw_video=False,
            record_wrench=record_wrench,     # aft 손목 F/T + 관절각 로깅
            wrench_topic=wrench_topic,
            record_doosan_force=record_doosan_force,   # doosan api 외력 로깅
            doosan_force_service=doosan_force_service,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager
        ) as env:
            cv2.setNumThreads(1)

            print("[INFO] Waiting for sensors to warm up...")
            time.sleep(2.0)

            # =================== 4. Warmup inference ====================
            print("[INFO] Warming up policy inference...")
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()
                if obs_pose_repr == 'relative':
                    obs_dict_np = get_real_relative_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                else:
                    obs_dict_np = get_real_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().cpu().numpy()
                assert action.shape[-1] == 9, \
                    f"Expected action dim=9, got {action.shape[-1]}"
                del result

            print("[INFO] Ready! Press 'S' to stop episode, Ctrl+C to quit.")

            # =================== 5. Control loop =======================
            while True:
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start      = time.monotonic() + start_delay

                    env.start_episode(eval_t_start)
                    frame_latency = 1.0 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("[INFO] Episode started!")

                    iter_idx = 0
                    correction_active = False
                    episode_kept = False   # S/timeout=저장(True), D=폐기(False)
                    while True:
                        # --- timing ---
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # --- get obs ---
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f"[INFO] Obs latency: {time.time() - obs_timestamps[-1]:.4f}s")

                        # --- inference ---
                        with torch.no_grad():
                            t_inf = time.time()
                            if obs_pose_repr == 'relative':
                                obs_dict_np = get_real_relative_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            else:
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # action: [Horizon, 9]  (pos(3) + rot6d(6), abs)
                            action = result['action'][0].detach().cpu().numpy()
                            print(f"[DEBUG] AI Action[0] (pos): {action[0, :3]}")
                            print(f"[DEBUG] AI Action[-1] (pos): {action[-1, :3]}")

                            # ---- orientation 디버그 (오른팔) ----
                            # action[:, 3:9] = rot6d (abs). 컨트롤러가 이걸 rotvec->ZYX euler(deg)로 보냄.
                            try:
                                cmd_rotvec0 = rot6d_to_rotvec(action[0, 3:9])
                                cmd_euler0 = R.from_rotvec(cmd_rotvec0).as_euler('ZYX', degrees=True)
                                print(f"[DEBUG] AI Action[0] rot6d  : {np.round(action[0, 3:9], 4)}")
                                print(f"[DEBUG] AI Action[0] rotvec : {np.round(cmd_rotvec0, 4)}")
                                print(f"[DEBUG] AI Action[0] euler ZYX(deg): {np.round(cmd_euler0, 2)}")
                                # 현재 팔 자세 (obs quat -> euler) 와 비교
                                curr_quat = np.asarray(obs['robot_quat_L'])[-1]  # [x,y,z,w]
                                curr_euler = R.from_quat(curr_quat).as_euler('ZYX', degrees=True)
                                print(f"[DEBUG] Current obs quat_L  : {np.round(curr_quat, 4)}")
                                print(f"[DEBUG] Current obs euler ZYX(deg): {np.round(curr_euler, 2)}")
                                print(f"[DEBUG] euler delta (cmd-curr): {np.round(cmd_euler0 - curr_euler, 2)}")
                            except Exception as _e:
                                print(f"[DEBUG] orientation debug skipped: {_e}")
                            # -------------------------------------

                            print(f"[INFO] Inference latency: {time.time() - t_inf:.4f}s")

                        # convert policy action to env actions
                        if delta_action:   # False
                            assert len(action) == 1
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            this_target_poses[:, :action.shape[-1]] = action
                        else:
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            this_target_poses[:, :action.shape[-1]] = action

                        # --- relative action -> abs target pose ---
                        # Keep env API unchanged: always execute absolute (pos+rot6d).
                        if action_pose_repr == 'relative':
                            try:
                                # Uses latest robot pose from env obs as reference.
                                print("Generated Action (Relative):", this_target_poses[0])
                                print(f"[DEBUG] Current robot abs pose_L: {obs['robot_pose_L'][-1]}")
                                print(f"[DEBUG] Current robot abs quat_L: {obs['robot_quat_L'][-1]}")
                                # Ensure we clip/check for absurdly large movements here
                                if np.any(np.abs(this_target_poses[..., :3]) > 0.5):
                                    print("WARNING: Unusually large relative translation detected!")

                                this_target_poses = get_abs_action_from_relative(
                                    action=this_target_poses,
                                    env_obs=obs
                                )
                                print("Converted Absolute Target Pose:", this_target_poses[0])
                            except Exception as e:
                                raise RuntimeError(
                                    "Failed to convert relative action to absolute. "
                                    "Check ckpt pose_repr/action definition and shape_meta. "
                                    f"Original error: {e}"
                                )

                        # deal with timing
                        # the same step actions are always the target for
                        # timing: already passed actions are removed
                        action_timestamps = (
                            np.arange(len(action), dtype=np.float64) + action_offset
                        ) * dt + obs_timestamps[-1]
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + 0.01)

                        if np.sum(is_new) == 0:
                            # 전부 지나버린 경우: 마지막 action이라도 실행
                            this_target_poses = this_target_poses[[-1]]
                            next_step_idx = int(
                                np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array(
                                [eval_t_start + next_step_idx * dt])
                            print("[WARN] Over budget! Executing last action.")
                        else:
                            this_target_poses = this_target_poses[is_new][:steps_per_inference]
                            action_timestamps  = action_timestamps[is_new][:steps_per_inference]

                        # --- keypress check (correction 토글 / 종료) ---
                        # C: correction 모드 토글. ON일 때 팔을 물리적으로 밀어 교정하면
                        #    해당 스텝이 stage=1로 기록되어, 이후
                        #    data_process/rush_replay_buffer_to_correction_hdf5.py 가
                        #    achieved-pose로 relabel 학습 데이터를 만든다.
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('c') or key_stroke == ord('C'):
                            correction_active = not correction_active
                            print(f"[INFO] Correction mode: "
                                  f"{'ON (human correcting)' if correction_active else 'OFF'}")

                        # --- execute ---
                        stages = np.full(len(this_target_poses),
                                         1 if correction_active else 0, dtype=np.int64)
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            stages=stages)
                        print(f"[INFO] Submitted {len(this_target_poses)} action steps. "
                              f"correction={correction_active}")

                        if key_stroke == ord('s') or key_stroke == ord('S'):
                            env.end_episode()
                            episode_kept = True
                            print("[INFO] Episode stopped by user (kept).")
                            break
                        if key_stroke == ord('d') or key_stroke == ord('D'):
                            env.drop_episode()
                            print("[INFO] Episode stopped by user (DISCARDED).")
                            break

                        # --- timeout check ---
                        if time.monotonic() - t_start > max_duration:
                            env.end_episode()
                            episode_kept = True
                            print("[INFO] Episode terminated by timeout.")
                            break

                        # --- wait for next cycle ---
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("\n[INFO] Interrupted! Stopping robot.")
                    env.end_episode()
                    if plot_force:
                        _safe_write_eval_diagnostics(
                            env.timeseries_hdf5_dir, output, steps_per_inference)
                    break

                # 저장된 에피소드면 시각화(힘 PNG + 정책 action 3D HTML) 저장
                if plot_force and episode_kept:
                    _safe_write_eval_diagnostics(
                        env.timeseries_hdf5_dir, output, steps_per_inference)

                # 에피소드 끝나면 다음 에피소드 대기
                print("[INFO] Press Enter to start next episode, or Ctrl+C to quit.")
                try:
                    import sys
                    sys.stdin.readline()   # 엔터 대기 (builtin input() 과 click 'input' 파라미터 충돌 방지)
                except (KeyboardInterrupt, EOFError):
                    break

    print("[INFO] Done.")


# %%
if __name__ == '__main__':
    main()
