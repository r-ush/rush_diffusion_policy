#!/usr/bin/env python3
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    return np.atleast_1d(data)


def _cols(data, prefix):
    names = data.dtype.names or ()
    return [name for name in names if name.startswith(prefix)]


def _time(data):
    t = np.asarray(data['wall_time'], dtype=np.float64)
    return t - t[0]


def _latest(path, suffix):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, f'*_{suffix}.csv')))
        if not files:
            raise FileNotFoundError(f'No *_{suffix}.csv under {path}')
        return files[-1]
    return path


def plot_schedule(schedule_path, out_dir):
    data = _load_csv(schedule_path)
    t = _time(data)
    raw_cols = _cols(data, 'raw_action_')
    target_cols = _cols(data, 'target_pose_')

    raw = np.stack([data[c] for c in raw_cols], axis=-1)
    target = np.stack([data[c] for c in target_cols], axis=-1)

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)

    axes[0].plot(t, raw[:, 0:3])
    axes[0].set_ylabel('xyz [m]')
    axes[0].legend(['x', 'y', 'z'], loc='upper left')
    axes[0].grid(True)

    axes[1].plot(t, raw[:, 3:9])
    axes[1].set_ylabel('raw rot6d')
    axes[1].grid(True)

    axes[2].plot(t, target[:, 3:6])
    axes[2].set_ylabel('rotvec [rad]')
    axes[2].legend(['rx', 'ry', 'rz'], loc='upper left')
    axes[2].grid(True)

    axes[3].plot(t, data['delta_pos'], label='delta_pos [m]')
    axes[3].plot(t, data['delta_rot'], label='delta_rot [rad]')
    axes[3].plot(t, data['delta_hand'], label='delta_hand')
    axes[3].set_ylabel('step jump')
    axes[3].legend(loc='upper left')
    axes[3].grid(True)

    axes[4].plot(t, data['rot6d_a1_norm'], label='a1_norm')
    axes[4].plot(t, data['rot6d_a2_norm'], label='a2_norm')
    axes[4].plot(t, data['rot6d_dot_unit'], label='unit_dot')
    axes[4].plot(t, data['rot6d_ortho_norm'], label='ortho_norm')
    axes[4].set_ylabel('rot6d quality')
    axes[4].set_xlabel('time [s]')
    axes[4].legend(loc='upper left')
    axes[4].grid(True)

    fig.suptitle(os.path.basename(schedule_path))
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'schedule_overview.png')
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_servo(servo_path, out_dir):
    data = _load_csv(servo_path)
    t = _time(data)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].semilogy(t, data['jac_cond'])
    axes[0].set_ylabel('Jacobian cond')
    axes[0].grid(True)

    axes[1].plot(t, data['dq_raw_norm'], label='dq_raw')
    axes[1].plot(t, data['dq_cmd_norm'], label='dq_cmd')
    axes[1].plot(t, data['dq_cmd_pos_norm'], label='dq_pos')
    axes[1].plot(t, data['dq_cmd_rot_norm'], label='dq_rot')
    axes[1].set_ylabel('dq norm')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    axes[2].plot(t, data['err_pos_norm'], label='err_pos [m]')
    axes[2].plot(t, data['err_rot_norm'], label='err_rot [rad]')
    axes[2].set_ylabel('servo error')
    axes[2].legend(loc='upper left')
    axes[2].grid(True)

    axes[3].plot(t, data['delta_pos'], label='delta_pos [m]')
    axes[3].plot(t, data['delta_rot'], label='delta_rot [rad]')
    axes[3].plot(t, data['delta_hand'], label='delta_hand')
    axes[3].set_ylabel('interpolated jump')
    axes[3].set_xlabel('time [s]')
    axes[3].legend(loc='upper left')
    axes[3].grid(True)

    fig.suptitle(os.path.basename(servo_path))
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'servo_ik.png')
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='action_debug directory or a *_schedule.csv path')
    parser.add_argument('--servo', default=None, help='optional *_servo.csv path')
    parser.add_argument('--out', default=None, help='output directory')
    args = parser.parse_args()

    schedule_path = _latest(args.path, 'schedule')
    if args.servo is None:
        servo_path = _latest(os.path.dirname(schedule_path), 'servo')
    else:
        servo_path = args.servo

    out_dir = args.out or os.path.dirname(schedule_path)
    os.makedirs(out_dir, exist_ok=True)

    print(plot_schedule(schedule_path, out_dir))
    print(plot_servo(servo_path, out_dir))


if __name__ == '__main__':
    main()
