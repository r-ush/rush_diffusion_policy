#!/home/vision/anaconda3/envs/robodiff/bin/python

# 실행코드
# python bae_eval_real_robot_rightarm_insert_plug.py --input data/outputs/260521_erase_board_unet_no_wrench/epoch\=0900-train_loss\=0.000.ckpt --output data/results
# python bae_eval_real_robot_rightarm_insert_plug.py --input data/outputs/260710_insert_box_hand_wrench_abs/epoch\=0900-train_loss\=0.001.ckpt --output data/results/260710_insert_box_hand --use_hand
"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
import signal
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.bae_real_env_rightarm_hand_insert_plug import DualarmRealEnv   # 새로 만듬
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    add_wrench_obs_noise,
    get_real_obs_resolution, 
    get_real_obs_dict,
    get_real_relative_obs_dict,
    get_abs_action_from_relative)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from analysis.modality_attribution.record_infer_obs import InferenceObsRecorder


OmegaConf.register_new_resolver("eval", eval, replace=True)


SQRT2 = np.sqrt(2.0) / 2.0
RIGHT_ROBOT_TO_WORLD = np.array(
    [
        [0.0, -SQRT2, -SQRT2],
        [-1.0, 0.0, 0.0],
        [0.0, SQRT2, -SQRT2],
    ],
    dtype=np.float64,
)
DEBUG_DIR_NAME = "eval_debug"
RIGHT_ARM_ACTION_DIM = 9
RIGHT_HAND_POLICY_DIM = 7


def _validate_hand_mode(shape_meta, use_hand):
    action_dim = int(shape_meta['action']['shape'][0])
    obs_meta = shape_meta['obs']
    hand_meta = obs_meta.get('hand_pose_R')
    hand_shape = None if hand_meta is None else tuple(hand_meta.get('shape', ()))

    expected_action_dim = RIGHT_ARM_ACTION_DIM + (
        RIGHT_HAND_POLICY_DIM if use_hand else 0
    )
    if action_dim != expected_action_dim:
        if not use_hand and action_dim == RIGHT_ARM_ACTION_DIM + RIGHT_HAND_POLICY_DIM:
            raise click.UsageError(
                'This checkpoint has a 7-DoF right-hand action. Re-run with --use_hand.'
            )
        raise click.UsageError(
            f'Hand mode={use_hand} expects action dim {expected_action_dim}, '
            f'but the checkpoint uses {action_dim}.'
        )

    if use_hand and hand_shape != (RIGHT_HAND_POLICY_DIM,):
        raise click.UsageError(
            f'--use_hand requires obs.hand_pose_R shape [{RIGHT_HAND_POLICY_DIM}], '
            f'but the checkpoint uses {hand_shape}.'
        )
    if not use_hand and hand_meta is not None:
        raise click.UsageError(
            'This checkpoint consumes hand_pose_R. Re-run with --use_hand.'
        )


def _right_robot_xyz_to_world(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz with last dim 3, got shape {xyz.shape}")
    return np.einsum("ij,...j->...i", RIGHT_ROBOT_TO_WORLD, xyz)


def _stack_debug(records, key, shape=None, dtype=np.float64):
    if not records:
        if shape is None:
            shape = (0,)
        return np.empty(shape, dtype=dtype)
    return np.asarray([record[key] for record in records], dtype=dtype)


def _latest_wrist_wrench_from_obs(obs):
    if "wrench_wrist_R" not in obs:
        return np.full((6,), np.nan, dtype=np.float64)
    wrench = np.asarray(obs["wrench_wrist_R"], dtype=np.float64)
    if wrench.size == 0:
        return np.full((6,), np.nan, dtype=np.float64)
    if wrench.ndim >= 2 and wrench.shape[-2] >= 6:
        return np.asarray(wrench[..., :6, -1], dtype=np.float64).reshape(-1, 6)[-1]
    if wrench.shape[-1] >= 6:
        return np.asarray(wrench[..., :6], dtype=np.float64).reshape(-1, 6)[-1]
    return np.full((6,), np.nan, dtype=np.float64)


def _episode_timeseries_hdf5_path(output_dir, episode_id):
    hdf5_dir = pathlib.Path(output_dir).joinpath("timeseries_hdf5")
    hdf5_path = hdf5_dir.joinpath(f"episode_{episode_id:06d}.hdf5")
    if hdf5_path.is_file():
        return hdf5_path
    partial_path = hdf5_dir.joinpath(f"episode_{episode_id:06d}_partial.hdf5")
    if partial_path.is_file():
        return partial_path
    return hdf5_path


def _nearest_indices(query_time, sample_time):
    query_time = np.asarray(query_time, dtype=np.float64)
    sample_time = np.asarray(sample_time, dtype=np.float64)
    if len(sample_time) == 0:
        return np.zeros((len(query_time),), dtype=np.int64)
    if len(sample_time) == 1:
        return np.zeros((len(query_time),), dtype=np.int64)
    insert = np.searchsorted(sample_time, query_time, side="left")
    insert = np.clip(insert, 1, len(sample_time) - 1)
    left = insert - 1
    right = insert
    use_right = np.abs(sample_time[right] - query_time) < np.abs(query_time - sample_time[left])
    return np.where(use_right, right, left)


def _load_wrist_ft_force_data(output_dir, episode_id):
    hdf5_path = _episode_timeseries_hdf5_path(output_dir, episode_id)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "wrist_ft" not in f or "wrench_wrist_R" not in f["wrist_ft"]:
            raise KeyError("wrist_ft/wrench_wrist_R")
        elapsed_s = np.asarray(f["wrist_ft/elapsed_s"], dtype=np.float64)
        wrench = np.asarray(f["wrist_ft/wrench_wrist_R"], dtype=np.float64)
        actual_elapsed_s = (
            np.asarray(f["actual/elapsed_s"], dtype=np.float64)
            if "actual" in f and "elapsed_s" in f["actual"]
            else np.empty((0,), dtype=np.float64)
        )
        actual_quat = (
            np.asarray(f["actual/robot_quat_R"], dtype=np.float64)
            if "actual" in f and "robot_quat_R" in f["actual"]
            else np.empty((0, 4), dtype=np.float64)
        )

    if len(elapsed_s) == 0 or len(wrench) == 0:
        raise ValueError("No wrist FT samples were recorded.")

    n_samples = min(len(elapsed_s), len(wrench))
    elapsed_s = elapsed_s[:n_samples]
    wrench_sensor = wrench[:n_samples]
    force_sensor = wrench_sensor[:, :3]
    force_world = None
    if len(actual_elapsed_s) > 0 and len(actual_quat) > 0:
        quat_idx = _nearest_indices(elapsed_s, actual_elapsed_s)
        tcp_rot_robot = st.Rotation.from_quat(actual_quat[quat_idx]).as_matrix()
        tcp_rot_world = np.einsum("ij,njk->nik", RIGHT_ROBOT_TO_WORLD, tcp_rot_robot)
        force_world = np.einsum("nij,nj->ni", tcp_rot_world, force_sensor)

    return elapsed_s, wrench_sensor, force_world


def _write_force_xyz_png(output_dir, episode_id, elapsed_s, force_xyz, frame_name, filename_suffix):
    elapsed_s = np.asarray(elapsed_s, dtype=np.float64)
    force_xyz = np.asarray(force_xyz, dtype=np.float64)
    if len(elapsed_s) == 0 or len(force_xyz) == 0:
        raise ValueError("No force samples to plot.")
    plot_time = elapsed_s - elapsed_s[0]

    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    debug_dir = pathlib.Path(output_dir).joinpath(DEBUG_DIR_NAME)
    debug_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    colors = ("#d62728", "#1f77b4", "#2ca02c")
    labels = ("Fx", "Fy", "Fz")
    for axis_index, (ax, color, label) in enumerate(zip(axes, colors, labels)):
        ax.plot(plot_time, force_xyz[:, axis_index], color=color, linewidth=1.8)
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"{label} (N)")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.01,
            0.86,
            label,
            color=color,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )

    axes[-1].set_xlabel("elapsed time (s)")
    fig.suptitle(f"Episode {episode_id:06d} wrist FT force xyz ({frame_name})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = debug_dir.joinpath(
        f"episode_{episode_id:06d}_wrist_ft_force{filename_suffix}_xyz.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"Wrist FT force PNG saved ({frame_name}): {png_path}")


def _write_wrist_ft_force_pngs(output_dir, episode_id):
    elapsed_s, wrench_sensor, force_world = _load_wrist_ft_force_data(output_dir, episode_id)
    _write_force_xyz_png(
        output_dir=output_dir,
        episode_id=episode_id,
        elapsed_s=elapsed_s,
        force_xyz=wrench_sensor[:, :3],
        frame_name="sensor/EE frame",
        filename_suffix="",
    )
    if force_world is None:
        print("[WARNING] Skipped world force PNG: missing actual/robot_quat_R.")
        return
    _write_force_xyz_png(
        output_dir=output_dir,
        episode_id=episode_id,
        elapsed_s=elapsed_s,
        force_xyz=force_world,
        frame_name="world frame",
        filename_suffix="_world",
    )


def _safe_write_wrist_ft_force_pngs(output_dir, episode_id):
    if episode_id is None:
        return
    try:
        _write_wrist_ft_force_pngs(output_dir, episode_id)
    except Exception as e:
        print(f"[WARNING] Failed to save wrist FT force PNGs: {e}")


def _load_actual_state(output_dir, episode_id, fallback_records):
    hdf5_path = _episode_timeseries_hdf5_path(output_dir, episode_id)
    if hdf5_path.is_file():
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            if "actual" in f and "elapsed_s" in f["actual"] and "robot_pose_R" in f["actual"]:
                return (
                    np.asarray(f["actual/elapsed_s"], dtype=np.float64),
                    np.asarray(f["actual/robot_pose_R"], dtype=np.float64),
                    np.asarray(f["actual/robot_quat_R"], dtype=np.float64)
                    if "robot_quat_R" in f["actual"]
                    else np.empty((0, 4), dtype=np.float64),
                )

    actual_time = _stack_debug(fallback_records, "elapsed_s")
    actual_pose = _stack_debug(fallback_records, "actual_pose", shape=(0, 3))
    actual_quat = _stack_debug(fallback_records, "actual_quat", shape=(0, 4))
    return actual_time, actual_pose, actual_quat


def _load_actual_traj(output_dir, episode_id, fallback_records):
    actual_time, actual_pose, _ = _load_actual_state(output_dir, episode_id, fallback_records)
    return actual_time, actual_pose


def _load_executed_action_targets(output_dir, episode_id, fallback_records):
    hdf5_path = _episode_timeseries_hdf5_path(output_dir, episode_id)
    if hdf5_path.is_file():
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            if (
                    "action_virtual_target" in f
                    and "elapsed_s" in f["action_virtual_target"]
                    and "timestamp" in f["action_virtual_target"]
                    and "action" in f["action_virtual_target"]):
                elapsed_s = np.asarray(
                    f["action_virtual_target/elapsed_s"],
                    dtype=np.float64,
                )
                action_timestamp = np.asarray(
                    f["action_virtual_target/timestamp"],
                    dtype=np.float64,
                )
                action = np.asarray(
                    f["action_virtual_target/action"],
                    dtype=np.float64,
                )
                if len(action) > 0:
                    n_samples = min(len(elapsed_s), len(action_timestamp), len(action))
                    elapsed_s = elapsed_s[:n_samples]
                    action_timestamp = action_timestamp[:n_samples]
                    action = action[:n_samples]
                    if fallback_records:
                        record_timestamps = _stack_debug(fallback_records, "action_timestamp")
                        if len(record_timestamps) > 0:
                            nearest = _nearest_indices(action_timestamp, record_timestamps)
                            keep = np.abs(record_timestamps[nearest] - action_timestamp) < 1e-4
                            if np.any(keep):
                                return elapsed_s[keep], action_timestamp[keep], action[keep]
                    return elapsed_s, action_timestamp, action

    return (
        _stack_debug(fallback_records, "elapsed_s"),
        _stack_debug(fallback_records, "action_timestamp"),
        _stack_debug(fallback_records, "action"),
    )


def _metadata_for_action_timestamps(action_timestamps, records, steps_per_inference):
    n_actions = len(action_timestamps)
    if n_actions == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    fallback_inference = (
        np.arange(n_actions, dtype=np.int64) // max(int(steps_per_inference), 1)
    )
    fallback_step = (
        np.arange(n_actions, dtype=np.int64) % max(int(steps_per_inference), 1)
    ) + 1
    fallback_raw = fallback_step - 1

    if not records:
        return fallback_inference, fallback_step, fallback_raw

    record_timestamps = _stack_debug(records, "action_timestamp")
    if len(record_timestamps) == 0:
        return fallback_inference, fallback_step, fallback_raw

    nearest = _nearest_indices(action_timestamps, record_timestamps)
    timestamp_error = np.abs(record_timestamps[nearest] - action_timestamps)
    valid = timestamp_error < 1e-4
    inference_index = fallback_inference.copy()
    step_in_inference = fallback_step.copy()
    raw_action_index = fallback_raw.copy()

    record_inference = _stack_debug(records, "inference_index", dtype=np.int64)
    record_step = _stack_debug(records, "step_in_inference", dtype=np.int64)
    record_raw = _stack_debug(records, "raw_action_index", dtype=np.int64)
    inference_index[valid] = record_inference[nearest[valid]]
    step_in_inference[valid] = record_step[nearest[valid]]
    raw_action_index[valid] = record_raw[nearest[valid]]
    return inference_index, step_in_inference, raw_action_index


def _sample_actual_pose_at_times(actual_time, actual_pose, query_elapsed_s, fallback_records):
    if len(query_elapsed_s) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(actual_time) > 0 and len(actual_pose) > 0:
        actual_idx = _nearest_indices(query_elapsed_s, actual_time)
        return np.asarray(actual_pose[actual_idx, :3], dtype=np.float64)
    return _stack_debug(fallback_records, "actual_pose", shape=(0, 3))


def _sample_actual_state_at_times(actual_time, actual_pose, actual_quat, query_elapsed_s, fallback_records):
    if len(query_elapsed_s) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 4), dtype=np.float64),
        )
    if len(actual_time) > 0 and len(actual_pose) > 0:
        actual_idx = _nearest_indices(query_elapsed_s, actual_time)
        actual_xyz = np.asarray(actual_pose[actual_idx, :3], dtype=np.float64)
        if len(actual_quat) > 0:
            quat_idx = np.clip(actual_idx, 0, len(actual_quat) - 1)
            actual_quat_xyzw = np.asarray(actual_quat[quat_idx, :4], dtype=np.float64)
        else:
            actual_quat_xyzw = np.full((len(query_elapsed_s), 4), np.nan, dtype=np.float64)
        return actual_xyz, actual_quat_xyzw

    actual_xyz = _stack_debug(fallback_records, "actual_pose", shape=(0, 3))
    actual_quat_xyzw = _stack_debug(fallback_records, "actual_quat", shape=(0, 4))
    if len(actual_xyz) != len(query_elapsed_s):
        actual_xyz = np.full((len(query_elapsed_s), 3), np.nan, dtype=np.float64)
    if len(actual_quat_xyzw) != len(query_elapsed_s):
        actual_quat_xyzw = np.full((len(query_elapsed_s), 4), np.nan, dtype=np.float64)
    return actual_xyz, actual_quat_xyzw


def _sample_wrist_wrench_at_times(output_dir, episode_id, query_elapsed_s, fallback_records):
    if len(query_elapsed_s) == 0:
        return np.empty((0, 6), dtype=np.float64)
    try:
        force_time, wrench_sensor, _ = _load_wrist_ft_force_data(output_dir, episode_id)
        if len(force_time) > 0 and len(wrench_sensor) > 0:
            force_idx = _nearest_indices(query_elapsed_s, force_time)
            return np.asarray(wrench_sensor[force_idx, :6], dtype=np.float64)
    except Exception:
        pass

    fallback_wrench = _stack_debug(fallback_records, "wrench_wrist_R", shape=(0, 6))
    if len(fallback_wrench) == len(query_elapsed_s):
        return fallback_wrench
    return np.full((len(query_elapsed_s), 6), np.nan, dtype=np.float64)


def _policy_action_debug_arrays(output_dir, episode_id, records, steps_per_inference):
    elapsed_s, action_timestamp, action = _load_executed_action_targets(
        output_dir,
        episode_id,
        records,
    )
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        if action.size == 0:
            action = np.empty((0, 0), dtype=np.float64)
        else:
            action = action.reshape(1, -1)
    inference_index, step_in_inference, raw_action_index = _metadata_for_action_timestamps(
        action_timestamp,
        records,
        steps_per_inference,
    )
    actual_time, actual_pose, actual_quat = _load_actual_state(output_dir, episode_id, records)
    actual_xyz, actual_quat_xyzw = _sample_actual_state_at_times(
        actual_time,
        actual_pose,
        actual_quat,
        elapsed_s,
        records,
    )
    wrench_wrist_R = _sample_wrist_wrench_at_times(
        output_dir,
        episode_id,
        elapsed_s,
        records,
    )
    return {
        "elapsed_s": elapsed_s,
        "timestamp": action_timestamp,
        "virtual_action": action,
        "actual_xyz": actual_xyz,
        "actual_quat_xyzw": actual_quat_xyzw,
        "wrench_wrist_R": wrench_wrist_R,
        "_inference_index": inference_index,
        "_step_in_inference": step_in_inference,
        "_raw_action_index": raw_action_index,
    }


def _rgb_obs_images_uint8(obs_dict_np, shape_meta):
    obs_shape_meta = shape_meta["obs"]
    if OmegaConf.is_config(obs_shape_meta):
        obs_shape_meta = OmegaConf.to_container(obs_shape_meta, resolve=True)

    images = {}
    for key, attr in obs_shape_meta.items():
        if attr.get("type", "low_dim") != "rgb" or key not in obs_dict_np:
            continue

        value = np.asarray(obs_dict_np[key])
        if value.ndim != 4:
            continue
        # Policy input image is TCHW. Store as THWC uint8 for easy inspection.
        if value.shape[1] in (1, 3, 4):
            value = np.moveaxis(value, 1, -1)
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
        elif value.dtype != np.uint8:
            value = np.clip(value, 0, 255).astype(np.uint8)
        images[key] = value
    return images


def _write_inference_images_group(hdf_file, image_records):
    if not image_records:
        return

    group = hdf_file.create_group("image")
    group.attrs["schema"] = (
        "Images are the RGB observations passed to policy inference, stored as "
        "uint8 THWC per inference. obs_timestamp/obs_elapsed_s have shape N x T."
    )
    group.create_dataset(
        "timestamp",
        data=np.asarray([record["obs_timestamps"] for record in image_records], dtype=np.float64),
    )
    group.create_dataset(
        "elapsed_s",
        data=np.asarray([record["obs_elapsed_s"] for record in image_records], dtype=np.float64),
    )

    image_group = group.create_group("rgb")
    image_keys = sorted({
        key
        for record in image_records
        for key in record["images"].keys()
    })
    image_group.attrs["keys"] = np.asarray(image_keys, dtype="S")
    for key in image_keys:
        values = [
            record["images"][key]
            for record in image_records
            if key in record["images"]
        ]
        if len(values) != len(image_records):
            print(f"[WARNING] Skipped image key {key}: missing in some inference records.")
            continue
        dataset = image_group.create_dataset(
            key,
            data=np.stack(values, axis=0),
            compression="gzip",
            compression_opts=4,
        )
        dataset.attrs["layout"] = "N_inference,T,H,W,C"
        dataset.attrs["dtype_range"] = "uint8 RGB 0..255"


def _write_pose_force_hdf5_datasets(hdf_file, arrays):
    hdf_file.attrs["schema"] = (
        "Minimal inference debug data: timestamp, image, actual pose, virtual pose, "
        "and wrist force/torque only."
    )
    hdf_file.create_dataset("timestamp", data=arrays["timestamp"])
    hdf_file.create_dataset("elapsed_s", data=arrays["elapsed_s"])

    actual_group = hdf_file.create_group("actual")
    actual_group.create_dataset("xyz", data=arrays["actual_xyz"], compression="gzip")
    actual_group.create_dataset(
        "quat_xyzw",
        data=arrays["actual_quat_xyzw"],
        compression="gzip",
    )

    virtual_action = np.asarray(arrays["virtual_action"], dtype=np.float64)
    if virtual_action.ndim == 1:
        virtual_action = virtual_action.reshape(0, 0) if virtual_action.size == 0 else virtual_action.reshape(1, -1)
    virtual_group = hdf_file.create_group("virtual")
    virtual_xyz = (
        virtual_action[:, :3]
        if virtual_action.shape[1] >= 3
        else np.full((len(virtual_action), 3), np.nan, dtype=np.float64)
    )
    virtual_group.create_dataset("xyz", data=virtual_xyz, compression="gzip")
    if virtual_action.shape[1] >= 9:
        virtual_group.create_dataset(
            "rot6d",
            data=virtual_action[:, 3:9],
            compression="gzip",
        )
    elif virtual_action.shape[1] > 3:
        virtual_group.create_dataset(
            "rotation",
            data=virtual_action[:, 3:],
            compression="gzip",
        )

    force_group = hdf_file.create_group("force")
    force_group.create_dataset(
        "wrench_wrist_R",
        data=arrays["wrench_wrist_R"],
        compression="gzip",
    )
    force_group.attrs["columns"] = np.asarray(
        ["fx", "fy", "fz", "tx", "ty", "tz"],
        dtype="S",
    )


def _write_policy_action_debug(output_dir, episode_id, records, steps_per_inference, image_records=None):
    if episode_id is None:
        return
    image_records = image_records or []

    output_dir = pathlib.Path(output_dir)
    debug_dir = output_dir.joinpath(DEBUG_DIR_NAME)
    debug_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = debug_dir.joinpath(f"episode_{episode_id:06d}_policy_targets.hdf5")
    html_path = debug_dir.joinpath(f"episode_{episode_id:06d}_policy_targets_3d.html")

    arrays = _policy_action_debug_arrays(output_dir, episode_id, records, steps_per_inference)
    virtual_action = np.asarray(arrays["virtual_action"], dtype=np.float64)
    has_actions = len(virtual_action) > 0
    import h5py
    with h5py.File(hdf5_path, "w") as f:
        _write_pose_force_hdf5_datasets(f, arrays)
        _write_inference_images_group(f, image_records)

    print(f"Policy action debug HDF5 saved: {hdf5_path}")
    if not has_actions:
        print("[WARNING] Skipped policy action 3D HTML: no executed actions were recorded.")
        return

    actual_time, actual_pose = _load_actual_traj(output_dir, episode_id, records)
    force_time = None
    force_world = None
    try:
        force_time, _, force_world = _load_wrist_ft_force_data(output_dir, episode_id)
    except Exception as e:
        print(f"[WARNING] Failed to load world force for 3D arrows: {e}")
    _write_policy_action_3d_html(
        html_path=html_path,
        episode_id=episode_id,
        elapsed_s=arrays["elapsed_s"],
        action=virtual_action,
        actual_time=actual_time,
        actual_pose=actual_pose,
        actual_pose_sampled=arrays["actual_xyz"],
        inference_index=arrays["_inference_index"],
        step_in_inference=arrays["_step_in_inference"],
        raw_action_index=arrays["_raw_action_index"],
        force_time=force_time,
        force_world=force_world,
    )
    print(f"Policy action 3D HTML saved: {html_path}")


def _safe_write_policy_action_debug(
        output_dir,
        episode_id,
        records,
        steps_per_inference,
        image_records=None):
    if episode_id is None:
        return
    try:
        _write_policy_action_debug(
            output_dir,
            episode_id,
            records,
            steps_per_inference,
            image_records=image_records,
        )
    except Exception as e:
        print(f"[WARNING] Failed to save policy action 3D visualization: {e}")


def _write_policy_action_3d_html(
        html_path,
        episode_id,
        elapsed_s,
        action,
        actual_time,
        actual_pose,
        actual_pose_sampled,
        inference_index,
        step_in_inference,
        raw_action_index,
        force_time=None,
        force_world=None):
    if len(action) == 0:
        return

    import plotly.graph_objects as go

    action_pos = _right_robot_xyz_to_world(action[:, :3])
    if len(actual_pose) > 0:
        actual_pos = _right_robot_xyz_to_world(actual_pose[:, :3])
    else:
        actual_pos = np.empty((0, 3), dtype=np.float64)
    if len(actual_pose_sampled) > 0:
        sampled_pos = _right_robot_xyz_to_world(actual_pose_sampled[:, :3])
    else:
        sampled_pos = np.empty((0, 3), dtype=np.float64)

    custom = np.column_stack(
        [
            np.arange(len(action), dtype=np.int64),
            inference_index,
            step_in_inference,
            raw_action_index,
            elapsed_s,
        ]
    )

    fig = go.Figure()
    if len(actual_pos) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=actual_pos[:, 0],
                y=actual_pos[:, 1],
                z=actual_pos[:, 2],
                mode="lines",
                name="real robot trajectory",
                line=dict(color="#111111", width=5),
                customdata=actual_time,
                hovertemplate=(
                    "real t=%{customdata:.3f}s<br>"
                    "world x=%{x:.4f} m<br>"
                    "world y=%{y:.4f} m<br>"
                    "world z=%{z:.4f} m<extra></extra>"
                ),
            )
        )

    actual_virtual_line_end = np.empty((0, 3), dtype=np.float64)
    if len(sampled_pos) == len(action_pos) and len(action_pos) > 0:
        line_x = []
        line_y = []
        line_z = []
        line_custom = []
        for index, (actual_point, virtual_point) in enumerate(zip(sampled_pos, action_pos)):
            if not (
                    np.all(np.isfinite(actual_point))
                    and np.all(np.isfinite(virtual_point))):
                continue
            line_x.extend([actual_point[0], virtual_point[0], None])
            line_y.extend([actual_point[1], virtual_point[1], None])
            line_z.extend([actual_point[2], virtual_point[2], None])
            line_custom.extend([index, index, index])
        if line_x:
            actual_virtual_line_end = action_pos
            fig.add_trace(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    name="actual to virtual",
                    line=dict(color="#2ca02c", width=4),
                    customdata=line_custom,
                    hovertemplate="cmd=%{customdata}<extra></extra>",
                )
            )

    fig.add_trace(
        go.Scatter3d(
            x=action_pos[:, 0],
            y=action_pos[:, 1],
            z=action_pos[:, 2],
            mode="lines+markers",
            name="policy target action",
            line=dict(color="#d62728", width=3),
            marker=dict(
                size=5,
                color="#d62728",
            ),
            customdata=custom,
            hovertemplate=(
                "cmd=%{customdata[0]:.0f}, infer=%{customdata[1]:.0f}, "
                "step=%{customdata[2]:.0f}, raw=%{customdata[3]:.0f}<br>"
                "t=%{customdata[4]:.3f}s<br>"
                "world x=%{x:.4f} m<br>"
                "world y=%{y:.4f} m<br>"
                "world z=%{z:.4f} m<extra></extra>"
            ),
        )
    )

    if len(sampled_pos) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=sampled_pos[:, 0],
                y=sampled_pos[:, 1],
                z=sampled_pos[:, 2],
                mode="markers",
                name="actual pose at inference",
                marker=dict(size=3.5, color="#1f77b4", opacity=0.65),
                customdata=elapsed_s,
                hovertemplate=(
                    "sampled actual t=%{customdata:.3f}s<br>"
                    "world x=%{x:.4f} m<br>"
                    "world y=%{y:.4f} m<br>"
                    "world z=%{z:.4f} m<extra></extra>"
                ),
            )
        )

    force_arrow_pos = np.empty((0, 3), dtype=np.float64)
    force_arrow_end = np.empty((0, 3), dtype=np.float64)
    if (
            force_time is not None
            and force_world is not None
            and len(actual_time) > 0
            and len(actual_pose) > 0
            and len(elapsed_s) > 0):
        actual_at_action_idx = _nearest_indices(elapsed_s, actual_time)
        force_at_action_idx = _nearest_indices(elapsed_s, force_time)
        force_arrow_pos = _right_robot_xyz_to_world(actual_pose[actual_at_action_idx, :3])
        force_vectors = np.asarray(force_world[force_at_action_idx], dtype=np.float64)
        valid_force = np.all(np.isfinite(force_arrow_pos), axis=1)
        valid_force &= np.all(np.isfinite(force_vectors), axis=1)
        valid_force &= np.linalg.norm(force_vectors, axis=1) > 1e-6
        force_arrow_pos = force_arrow_pos[valid_force]
        force_vectors = force_vectors[valid_force]

        if len(force_arrow_pos) > 0:
            max_arrows = 80
            if len(force_arrow_pos) > max_arrows:
                arrow_idx = np.unique(
                    np.linspace(0, len(force_arrow_pos) - 1, max_arrows, dtype=np.int64)
                )
                force_arrow_pos = force_arrow_pos[arrow_idx]
                force_vectors = force_vectors[arrow_idx]

            force_norm = np.linalg.norm(force_vectors, axis=1)
            scale_force = np.nanpercentile(force_norm, 95)
            if not np.isfinite(scale_force) or scale_force < 1e-6:
                scale_force = np.nanmax(force_norm)
            if not np.isfinite(scale_force) or scale_force < 1e-6:
                scale_force = 1.0

            force_arrow_unit_vectors = force_vectors / scale_force
            force_arrow_end = force_arrow_pos + force_arrow_unit_vectors * 0.035
            line_x = []
            line_y = []
            line_z = []
            for start, end in zip(force_arrow_pos, force_arrow_end):
                line_x.extend([start[0], end[0], None])
                line_y.extend([start[1], end[1], None])
                line_z.extend([start[2], end[2], None])

            fig.add_trace(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    name="world force arrows",
                    line=dict(color="#7e22ce", width=4),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Cone(
                    x=force_arrow_end[:, 0],
                    y=force_arrow_end[:, 1],
                    z=force_arrow_end[:, 2],
                    u=force_arrow_unit_vectors[:, 0],
                    v=force_arrow_unit_vectors[:, 1],
                    w=force_arrow_unit_vectors[:, 2],
                    customdata=force_vectors,
                    name="world force arrow heads",
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=0.012,
                    colorscale=[[0.0, "#7e22ce"], [1.0, "#7e22ce"]],
                    showscale=False,
                    hovertemplate=(
                        "world force arrow<br>"
                        "Fx=%{customdata[0]:.3f} N<br>"
                        "Fy=%{customdata[1]:.3f} N<br>"
                        "Fz=%{customdata[2]:.3f} N<extra></extra>"
                    ),
                )
            )

    replan_idx = np.flatnonzero(np.r_[True, np.diff(inference_index) != 0])
    if len(replan_idx) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=action_pos[replan_idx, 0],
                y=action_pos[replan_idx, 1],
                z=action_pos[replan_idx, 2],
                mode="markers",
                name="new inference boundary",
                marker=dict(
                    size=7,
                    color="#d62728",
                    symbol="diamond",
                    line=dict(color="#111111", width=1.5),
                ),
                customdata=custom[replan_idx],
                hovertemplate=(
                    "new inference boundary<br>"
                    "cmd=%{customdata[0]:.0f}, infer=%{customdata[1]:.0f}<br>"
                    "t=%{customdata[4]:.3f}s<extra></extra>"
                ),
            )
        )

    centers = [action_pos]
    if len(actual_pos) > 0:
        centers.append(actual_pos)
    if len(sampled_pos) > 0:
        centers.append(sampled_pos)
    if len(actual_virtual_line_end) > 0:
        centers.append(actual_virtual_line_end)
    if len(force_arrow_pos) > 0:
        centers.append(force_arrow_pos)
    if len(force_arrow_end) > 0:
        centers.append(force_arrow_end)
    centers = np.vstack(centers)
    center = centers.mean(axis=0)
    radius = max(float(np.ptp(centers, axis=0).max()) * 0.58, 0.01)

    fig.update_layout(
        title=f"Policy target actions, episode {episode_id:06d} (world frame)",
        template="plotly_white",
        width=1100,
        height=850,
        scene=dict(
            xaxis=dict(title="world x (m)", range=[center[0] - radius, center[0] + radius]),
            yaxis=dict(title="world y (m)", range=[center[1] - radius, center[1] + radius]),
            zaxis=dict(title="world z (m)", range=[center[2] - radius, center[2] + radius]),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)


def _safe_write_eval_diagnostics(
        output_dir,
        episode_id,
        action_records,
        steps_per_inference,
        image_records=None):
    _safe_write_policy_action_debug(
        output_dir,
        episode_id,
        action_records,
        steps_per_inference,
        image_records=image_records,
    )
    _safe_write_wrist_ft_force_pngs(output_dir, episode_id)


def _set_shutdown_signal_handlers(handler):
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handler)
    return old_handlers


def _restore_signal_handlers(old_handlers):
    for sig, handler in old_handlers.items():
        signal.signal(sig, handler)


def _finish_episode_and_save_diagnostics(
        env,
        output_dir,
        episode_id,
        action_records,
        steps_per_inference,
        image_records,
        reason,
        obs_recorder=None):
    if episode_id is None:
        return

    print(f"Finalizing episode {episode_id:06d} ({reason}); please wait for files to flush.")
    old_handlers = _set_shutdown_signal_handlers(signal.SIG_IGN)
    try:
        try:
            env.end_episode()
        except KeyboardInterrupt:
            print("[WARNING] Ignored Ctrl+C while finalizing episode.")
            try:
                env.end_episode()
            except Exception as e:
                print(f"[WARNING] Failed to end episode after interrupt: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to end episode cleanly: {e}")

        _safe_write_eval_diagnostics(
            output_dir,
            episode_id,
            action_records,
            steps_per_inference,
            image_records=image_records,
        )
        if obs_recorder is not None:
            try:
                obs_recorder.save(output_dir, episode_id)
            except Exception as e:
                print(f"[WARNING] Failed to save inference obs snapshots: {e}")
        print(f"Episode {episode_id:06d} finalize complete.")
    finally:
        _restore_signal_handlers(old_handlers)


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')   # checkpoint
@click.option('--output', '-o', required=True, help='Directory to save recording')   
@click.option('--robot_ip', '-ri', default="192.168.111.50", required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')   
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")   # 몇개의 action 실행할건지
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")  
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--wrench_noise_force_mean', '--wrench-noise-force-mean', default=0.0, type=float, show_default=True, help='Constant offset added to policy wrench force channels in N.')
@click.option('--wrench_noise_force_uniform_min', '--wrench-noise-force-uniform-min', default=None, type=float, help='Minimum uniform noise added to policy wrench force channels in N.')
@click.option('--wrench_noise_force_uniform_max', '--wrench-noise-force-uniform-max', default=None, type=float, help='Maximum uniform noise added to policy wrench force channels in N.')
@click.option('--wrench_noise_force_std', '--wrench-noise-force-std', default=0.0, type=float, show_default=True, help='Gaussian noise std added to policy wrench force channels in N. <=0 disables force noise.')
@click.option('--wrench_noise_torque_mean', '--wrench-noise-torque-mean', default=0.0, type=float, show_default=True, help='Constant offset added to policy wrist torque channels in Nm.')
@click.option('--wrench_noise_torque_uniform_min', '--wrench-noise-torque-uniform-min', default=None, type=float, help='Minimum uniform noise added to policy wrist torque channels in Nm.')
@click.option('--wrench_noise_torque_uniform_max', '--wrench-noise-torque-uniform-max', default=None, type=float, help='Maximum uniform noise added to policy wrist torque channels in Nm.')
@click.option('--wrench_noise_torque_std', '--wrench-noise-torque-std', default=0.0, type=float, show_default=True, help='Gaussian noise std added to policy wrist torque channels in Nm. <=0 disables torque noise.')
@click.option('--wrench_noise_seed', '--wrench-noise-seed', default=None, type=int, help='Optional RNG seed for policy wrench observation noise.')
@click.option('--record_wrist_wrench/--no_record_wrist_wrench', default=True, help="Include right wrist FT in the episode HDF5 timeseries.")
@click.option(
    '--use_hand/--no_use_hand',
    default=False,
    help='Enable 7-DoF right-hand observation and action control.',
)
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, wrench_noise_force_mean, wrench_noise_force_std,
    wrench_noise_force_uniform_min, wrench_noise_force_uniform_max,
    wrench_noise_torque_mean, wrench_noise_torque_std,
    wrench_noise_torque_uniform_min, wrench_noise_torque_uniform_max,
    wrench_noise_seed, record_wrist_wrench, use_hand):

    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt

    _set_shutdown_signal_handlers(_raise_keyboard_interrupt)

    # load checkpoint; checkpoint의 cfg 및 파라미터들 다 가져옴
    ckpt_path = pathlib.Path(input)
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']   # yaml에 있던 변수들 설정값
    _validate_hand_mode(cfg.task.shape_meta, use_hand)
    
    # Head = 242422304502, Front = 336222070518, Left = 218622276386, Right = 126122270712
    # serial_numbers = ['126122270712', '151222078010'] # right, table
    serial_numbers = ['126122270712'] # right
   
    # RTC
    use_pigdm = False
    if use_pigdm == True:
        cfg._target_ = 'diffusion_policy.workspace.bae_train_diffusion_unet_hybrid_pigdm_workspace.TrainDiffusionUnetHybridPigdmWorkspace'
        cfg.policy._target_ = "diffusion_policy.policy.bae_diffusion_unet_hybrid_image_policy_pigdm.DiffusionUnetHybridImagePigdmPolicy"
        cfg.policy.noise_scheduler._target_ = "bae_scheduling_ddim_pigdm.DDIMPIGDMScheduler"

    cls = hydra.utils.get_class(cfg._target_)   # WorkSpace 설정
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # 여기서 workspace.model에 cfg.policy가 들어감


    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False  
    workspace_target = str(OmegaConf.select(cfg, "_target_", default="")).lower()
    policy_target = str(OmegaConf.select(cfg, "policy._target_", default="")).lower()
    is_diffusion_policy = any(
        'diffusion' in target
        for target in (str(cfg.name).lower(), workspace_target, policy_target)
    )
    if is_diffusion_policy:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model   # state_dicts의 model을 가져옴 (가중치 값들)
        if cfg.training.use_ema:
            policy = workspace.ema_model   # ema_model 가져옴 (가중치 값들)
        device = torch.device('cuda')
        policy.eval().to(device)
        
        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations; 노이즈 제거 step 수
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1   # 과거부터 horizon 뽑고, obs만큼 빼고, 1 더하기 (16 - 2 + 1 = 15)

    else:
        raise RuntimeError(
            "Unsupported policy type: ",
            cfg.name,
            cfg._target_,
            OmegaConf.select(cfg, "policy._target_", default=None))


    # setup experiment
    dt = 1/frequency
    wrench_noise_rng = np.random.default_rng(wrench_noise_seed)

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # obs의 image 해상도 (width, height)
    n_obs_steps = cfg.n_obs_steps   # obs 관측 step 수
    print("n_obs_steps: ", n_obs_steps)   # obs 관측 step수 (2)
    print("steps_per_inference:", steps_per_inference)   # 예측한 action sequence에서 몇개의 action 실행할건지 (6)
    print("action_offset:", action_offset)   # action 지연 실행 (0)
    print("timeseries_hdf5_dir:", pathlib.Path(output).joinpath('timeseries_hdf5'))
    print("wrench noise force mean N:", wrench_noise_force_mean)
    print("wrench noise force uniform N:", wrench_noise_force_uniform_min, wrench_noise_force_uniform_max)
    print("wrench noise force std N:", wrench_noise_force_std)
    print("wrench noise torque mean Nm:", wrench_noise_torque_mean)
    print("wrench noise torque uniform Nm:", wrench_noise_torque_uniform_min, wrench_noise_torque_uniform_max)
    print("wrench noise torque std Nm:", wrench_noise_torque_std)
    print("wrench noise seed:", wrench_noise_seed)
    print("right hand control:", "enabled" if use_hand else "disabled")


    # =============== relative ==================
    # 있으면 'relative' or 'abs' / 없으면 None
    obs_pose_repr = OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default=None)
    action_pose_repr = OmegaConf.select(cfg, "task.pose_repr.action_pose_repr", default=None)

    # ===========================================

    # sharedmemory에 데이터들 쌓기; 같은 공유 공간 사용
    with SharedMemoryManager() as shm_manager:
        with DualarmRealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,   
            camera_serial_numbers=serial_numbers,
            n_obs_steps=n_obs_steps,   
            shape_meta=cfg.task.shape_meta,
            obs_image_resolution=obs_res, # (224,224)
            obs_float32=True,   
            init_joints=init_joints,   # False
            enable_multi_cam_vis=True,   # 별도 프로세스에서 policy 입력 해상도 image 시각화
            record_raw_video=False,   # 원본 화질 영상 저장 
            record_wrist_wrench=record_wrist_wrench,
            use_hand=use_hand,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)


            # Realsense-viewer에서 설정
            # Should be the same as demo
            # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            
            # obs 받아오기
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()

                # 받은 obs에서 image 정규화 및 다듬기, pose 다듬기
                # obs: relative or abs
                if obs_pose_repr == 'relative':
                    obs_dict_np = get_real_relative_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                else:
                    obs_dict_np = get_real_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict_np = add_wrench_obs_noise(
                    obs_dict_np,
                    cfg.task.shape_meta,
                    rng=wrench_noise_rng,
                    force_mean=wrench_noise_force_mean,
                    force_uniform_min=wrench_noise_force_uniform_min,
                    force_uniform_max=wrench_noise_force_uniform_max,
                    force_std=wrench_noise_force_std,
                    torque_mean=wrench_noise_torque_mean,
                    torque_uniform_min=wrench_noise_torque_uniform_min,
                    torque_uniform_max=wrench_noise_torque_uniform_max,
                    torque_std=wrench_noise_torque_std,
                )

                for key in obs_dict_np.keys():
                    print(f"{key}: {obs_dict_np[key].shape}, {obs_dict_np[key].dtype}")

                # shape_meta 계층구조는 유지하면서 np --> tensor로 변환, 텐서 배치차원 추가
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                # obs로 action 예측 
                if use_pigdm == True:
                    result = policy.predict_action_pigdm(obs_dict, obs)
                else:
                    result = policy.predict_action(obs_dict)


                # 실제 실행할 action trajectory
                action = result['action'][0].detach().to('cpu').numpy()   # [0]은 배치차원 제거, tensor --> np
                assert action.shape[-1] == cfg.task.shape_meta.action.shape[0]   # action 차원에 맞게 바꿔주기
                del result

            np.set_printoptions(suppress=True, floatmode="fixed", precision=11)
            print('Ready!')
            while True:
                
                # ========== policy control loop ==============
                episode_id = None
                action_debug_records = []
                image_debug_records = []
                obs_recorder = InferenceObsRecorder()
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay   # 시스템시간, 영상 로그용
                    t_start = time.monotonic() + start_delay   # 로봇 제어 시간
                    episode_id = env.replay_buffer.n_episodes
                    action_debug_records = []
                    image_debug_records = []
                    obs_recorder = InferenceObsRecorder()
                    # print("[TIME] t_start: ", t_start%100)

                    env.start_episode(eval_t_start)   # 영상 저장 시작
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency; 카메라 프레임 잘 받아오도록
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0   # trajectory 실행 개수
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:  
                        # calculate timing; 실행할 action 만큼 기다릴 시간
                        # print("[TIME] current time: ", time.monotonic()%100)
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
                        # print("[TIME] t_cycle_end: ", t_cycle_end%100)

                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference; action 예측
                        with torch.no_grad():
                            s = time.time()

                            # obs: relative or abs
                            if obs_pose_repr == 'relative':
                                obs_dict_np = get_real_relative_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            else:
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict_np = add_wrench_obs_noise(
                                obs_dict_np,
                                cfg.task.shape_meta,
                                rng=wrench_noise_rng,
                                force_mean=wrench_noise_force_mean,
                                force_uniform_min=wrench_noise_force_uniform_min,
                                force_uniform_max=wrench_noise_force_uniform_max,
                                force_std=wrench_noise_force_std,
                                torque_mean=wrench_noise_torque_mean,
                                torque_uniform_min=wrench_noise_torque_uniform_min,
                                torque_uniform_max=wrench_noise_torque_uniform_max,
                                torque_std=wrench_noise_torque_std,
                            )

                            inference_index = (
                                int(iter_idx // steps_per_inference)
                                if steps_per_inference > 0 else int(iter_idx)
                            )
                            inference_images = _rgb_obs_images_uint8(
                                obs_dict_np,
                                cfg.task.shape_meta,
                            )
                            if inference_images:
                                image_debug_records.append({
                                    "inference_index": inference_index,
                                    "obs_timestamps": np.asarray(
                                        obs_timestamps,
                                        dtype=np.float64,
                                    ),
                                    "obs_elapsed_s": np.asarray(
                                        obs_timestamps - eval_t_start,
                                        dtype=np.float64,
                                    ),
                                    "images": inference_images,
                                })
                       
                            obs_recorder.add(
                                inference_index,
                                obs_dict_np,
                                obs_timestamps,
                                eval_t_start,
                            )

                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                            # action
                            if use_pigdm == True:
                                result = policy.predict_action_pigdm(obs_dict, obs)
                            else:
                                result = policy.predict_action(obs_dict)


                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()   # 실행할 action[Horizon, Action_Dim]
                            
                            # action: relative or abs
                            if action_pose_repr == 'relative':
                                action = get_abs_action_from_relative(action=action, env_obs=obs)

                            print('Inference latency:', time.time() - s)

                        this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                        this_target_poses[:, :action.shape[-1]] = action

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)   # 현재시점 이후 action만 실행
                        # print("[DEBUG] action_timestamps: ", np.array(action_timestamps)%10)
                        # print("[DEBUG] is_new: ", is_new)

                        ############################################ timestamp
                        # while문은 6 * 0.1 주기로 무조건 돌음
                        # 현재시간 t / 이용하는 obs_timestamp = t - obs_latency
                        #   1           2              3      4      5      6      7         8         9     10    11    12    13    14    15 16
                        # -0.1    obs_timestamp      +0.1   +0.2   +0.3   +0.4   +0.5      +0.6
                        #                                                        -0.1  obs_timestamp  +0.1  +0.2  +0.3  +0.4  +0.5  +0.6
                        # print("Current time:", curr_time)
                        # print("Action timestamps:", action_timestamps)
                        ############################################
                        
                        if np.sum(is_new) == 0:   # 전부 지나버림
                            print('[WARNING] All actions are outdated!')
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]   # 마지막 action이라도 실행
                            selected_raw_action_indices = np.array(
                                [len(action) - 1],
                                dtype=np.int64,
                            )
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])

                        else:   # is_new = 1 인것만 실행
                            this_target_poses = this_target_poses[is_new]#[:6]
                            action_timestamps = action_timestamps[is_new]#[:6]
                            selected_raw_action_indices = np.flatnonzero(is_new).astype(np.int64)

                        if "robot_pose_R" in obs and len(obs["robot_pose_R"]) > 0:
                            actual_pose_sample = np.asarray(
                                obs["robot_pose_R"][-1],
                                dtype=np.float64,
                            )
                        else:
                            actual_pose_sample = np.full((3,), np.nan, dtype=np.float64)
                        if "robot_quat_R" in obs and len(obs["robot_quat_R"]) > 0:
                            actual_quat_sample = np.asarray(
                                obs["robot_quat_R"][-1],
                                dtype=np.float64,
                            )
                        else:
                            actual_quat_sample = np.full((4,), np.nan, dtype=np.float64)
                        wrench_wrist_sample = _latest_wrist_wrench_from_obs(obs)
                        debug_count = len(this_target_poses)
                        if steps_per_inference > 0:
                            debug_count = min(debug_count, int(steps_per_inference))
                        for target_pose, target_timestamp, raw_action_index in zip(
                                this_target_poses[:debug_count],
                                action_timestamps[:debug_count],
                                selected_raw_action_indices[:debug_count]):
                            action_debug_records.append({
                                "elapsed_s": float(target_timestamp - eval_t_start),
                                "action_timestamp": float(target_timestamp),
                                "inference_index": inference_index,
                                "step_in_inference": int(raw_action_index) + 1,
                                "raw_action_index": int(raw_action_index),
                                "action": np.asarray(target_pose, dtype=np.float64),
                                "actual_pose": actual_pose_sample,
                                "actual_quat": actual_quat_sample,
                                "wrench_wrist_R": wrench_wrist_sample,
                            })

                        # execute actions; 실제 action 실행부분; 
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")
                       

                        # 's' 누르면 종료
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            _finish_episode_and_save_diagnostics(
                                env=env,
                                output_dir=output,
                                episode_id=episode_id,
                                action_records=action_debug_records,
                                steps_per_inference=steps_per_inference,
                                image_records=image_debug_records,
                                reason="operator stop",
                                obs_recorder=obs_recorder,
                            )
                            print('Stopped.')
                            break


                        # auto termination; 한계시간 지나면 종료
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            _finish_episode_and_save_diagnostics(
                                env=env,
                                output_dir=output,
                                episode_id=episode_id,
                                action_records=action_debug_records,
                                steps_per_inference=steps_per_inference,
                                image_records=image_debug_records,
                                reason="timeout",
                                obs_recorder=obs_recorder,
                            )
                            break


                        # wait for execution; 로봇이 action 여러개 실행할동안 기다림
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference
                        # print("[TIME] cycle 끝 시간: ", time.monotonic()%100, time.time()%10)
                        # time.sleep(1)

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    _finish_episode_and_save_diagnostics(
                        env=env,
                        output_dir=output,
                        episode_id=episode_id,
                        action_records=action_debug_records,
                        steps_per_inference=steps_per_inference,
                        image_records=image_debug_records,
                        reason="keyboard interrupt",
                        obs_recorder=obs_recorder,
                    )
                    print("Stopped.")
                    return
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
