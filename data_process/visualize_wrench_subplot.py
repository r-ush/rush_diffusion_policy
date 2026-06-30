"""
Wrench Subplot Visualizer (Interactive)
- 4개의 demo subplot을 고정 표시 (항상 4행 × 2열)
- 좌/우 방향키 또는 Prev/Next 버튼으로 페이지 이동
- 각 demo에 대해 Fx, Fy, Fz, Mx, My, Mz (6축 wrench) 시각화
- demo 부족 시 빈 subplot 유지

Usage:
  python visualize_wrench_subplot.py --file <path_to_hdf5>
  python visualize_wrench_subplot.py --file <path_to_hdf5> --wrench_key wrench_wrist_R

Controls:
  → / Next 버튼 : 다음 4개 demo
  ← / Prev 버튼 : 이전 4개 demo
  q             : 종료
"""

import os
import sys
import glob
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


DEMOS_PER_PAGE = 4


def demo_sort_key(name):
    """demo_10, demo_2 같은 이름을 숫자 기준으로 정렬하기 위한 키."""
    try:
        return int(name.split('_')[-1])
    except (ValueError, IndexError):
        return name


def find_latest_hdf5(base_dir='/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7'):
    files = glob.glob(os.path.join(base_dir, '*/*.hdf5'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_wrench_key(obs):
    """HDF5 observations에서 사용 가능한 wrench 키를 자동 탐색"""
    candidates = [
        'wrench_wrist_R',
        'F_e_raw',
        'wrench_thumb_R',
        'wrench_index_R',
        'wrench_middle_R',
        'wrench_ring_R',
        'wrench_baby_R',
    ]
    for key in candidates:
        if key in obs:
            return key
    for key in obs.keys():
        if 'wrench' in key.lower() or 'force' in key.lower():
            return key
    return None


def plot_wrench_subplots(filepath, wrench_key=None):
    if filepath is None:
        filepath = find_latest_hdf5()
    if filepath is None or not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    print(f"\n[INFO] Loading: {filepath}")

    f = h5py.File(filepath, 'r')

    if 'data' not in f:
        print("Error: No 'data' group found.")
        f.close()
        return

    data_group = f['data']
    demos = sorted(data_group.keys(), key=demo_sort_key)
    if not demos:
        print("Error: No demos found.")
        f.close()
        return

    total_demos = len(demos)
    total_pages = max(1, (total_demos + DEMOS_PER_PAGE - 1) // DEMOS_PER_PAGE)
    print(f"[INFO] Found {total_demos} demo(s), {total_pages} page(s)")

    # ── 색상 / 라벨 ──
    force_labels = ['Fx', 'Fy', 'Fz']
    torque_labels = ['Mx', 'My', 'Mz']
    force_colors = ['#e63946', '#457b9d', '#2a9d8f']
    torque_colors = ['#e76f51', '#264653', '#a8dadc']

    # ── Figure 생성 (4행 × 2열 고정) ──
    fig, axes = plt.subplots(DEMOS_PER_PAGE, 2, figsize=(18, 18), squeeze=False)
    fig.subplots_adjust(bottom=0.07, top=0.93, hspace=0.35)

    state = {'page': 0}

    def draw_page(page):
        """현재 페이지의 demo들을 그린다."""
        state['page'] = page
        start = page * DEMOS_PER_PAGE

        # 모든 subplot 초기화
        for row in range(DEMOS_PER_PAGE):
            for col in range(2):
                axes[row, col].cla()
                axes[row, col].grid(True, alpha=0.3)

        for row in range(DEMOS_PER_PAGE):
            didx = start + row

            if didx >= total_demos:
                # demo 부족: 빈 subplot
                axes[row, 0].set_title(f"(empty)", fontsize=10, color='gray')
                axes[row, 1].set_title(f"(empty)", fontsize=10, color='gray')
                axes[row, 0].set_facecolor('#fafafa')
                axes[row, 1].set_facecolor('#fafafa')
                continue

            demo_name = demos[didx]
            obs = data_group[demo_name]['observations']

            # wrench 키 결정
            wkey = wrench_key if wrench_key and wrench_key in obs else find_wrench_key(obs)
            if wkey is None:
                axes[row, 0].set_title(f"{demo_name} – No wrench data", color='red')
                axes[row, 1].set_title(f"{demo_name} – No wrench data", color='red')
                continue

            wrench = np.array(obs[wkey])
            n_cols = wrench.shape[1] if wrench.ndim == 2 else 0

            # 타임스탬프
            t = None
            for tkey in ['timestamp_wrench', 'timestamp_robot', 'timestamp']:
                if tkey in obs:
                    t = np.array(obs[tkey]).flatten()
                    break
            if t is None or len(t) != len(wrench):
                t = np.arange(len(wrench))

            x_label = "Time (s)" if t[-1] > len(wrench) * 0.5 else "Step"

            # Force
            ax_f = axes[row, 0]
            for i, (label, color) in enumerate(zip(force_labels, force_colors)):
                if i < n_cols:
                    ax_f.plot(t, wrench[:, i], label=label, color=color,
                              linewidth=1.0, alpha=0.85)
            ax_f.set_title(f"{demo_name} – Force", fontsize=11, fontweight='bold')
            ax_f.set_ylabel("Force (N)")
            ax_f.legend(loc='upper right', fontsize=8, ncol=3)
            ax_f.grid(True, alpha=0.3)

            # Torque
            ax_t = axes[row, 1]
            for i, (label, color) in enumerate(zip(torque_labels, torque_colors)):
                col_idx = 3 + i
                if col_idx < n_cols:
                    ax_t.plot(t, wrench[:, col_idx], label=label, color=color,
                              linewidth=1.0, alpha=0.85)
            ax_t.set_title(f"{demo_name} – Torque", fontsize=11, fontweight='bold')
            ax_t.set_ylabel("Torque (Nm)")
            ax_t.legend(loc='upper right', fontsize=8, ncol=3)
            ax_t.grid(True, alpha=0.3)

            # 마지막 행에 x-label
            if row == DEMOS_PER_PAGE - 1 or didx == total_demos - 1:
                ax_f.set_xlabel(x_label)
                ax_t.set_xlabel(x_label)

        fig.suptitle(
            f"Wrench  –  {os.path.basename(filepath)}    "
            f"[Page {page + 1}/{total_pages}]    "
            f"(demos {start}–{min(start + DEMOS_PER_PAGE, total_demos) - 1})",
            fontsize=13, fontweight='bold'
        )
        fig.canvas.draw_idle()

    # ── 네비게이션 버튼 ──
    ax_prev = fig.add_axes([0.35, 0.01, 0.1, 0.03])
    ax_next = fig.add_axes([0.55, 0.01, 0.1, 0.03])
    btn_prev = Button(ax_prev, '◀  Prev')
    btn_next = Button(ax_next, 'Next  ▶')

    def go_next(event=None):
        if state['page'] < total_pages - 1:
            draw_page(state['page'] + 1)

    def go_prev(event=None):
        if state['page'] > 0:
            draw_page(state['page'] - 1)

    btn_next.on_clicked(go_next)
    btn_prev.on_clicked(go_prev)

    # ── 키보드 네비게이션 ──
    def on_key(event):
        if event.key == 'right':
            go_next()
        elif event.key == 'left':
            go_prev()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ── 초기 페이지 그리기 ──
    draw_page(0)
    plt.show()

    f.close()
    print("\n[INFO] Viewer closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wrench subplot visualizer (interactive)")
    parser.add_argument('--file', type=str, default=None,
                        help="HDF5 file path. If omitted, uses the latest file.")
    parser.add_argument('--wrench_key', type=str, default=None,
                        help="Specific wrench key to use (e.g. wrench_wrist_R, F_e_raw)")
    args = parser.parse_args()
    plot_wrench_subplots(args.file, args.wrench_key)
