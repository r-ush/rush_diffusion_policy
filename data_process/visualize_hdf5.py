"""
HDF5 Data Viewer
- OpenCV 창: 이미지 재생 + 프레임 트랙바 (즉각 반응)
- Matplotlib 창: Joint / Wrench 전체 그래프 + 프레임 위치 마커
"""
import os
import sys
import glob
import h5py
import numpy as np
import cv2
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────
# HDF5 파일 찾기
# ──────────────────────────────────────────────
# def find_latest_hdf5(base_dir='/media/vision/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7'):
#     files = glob.glob(os.path.join(base_dir, '*/*.hdf5'))
#     if not files:
#         return None
#     return max(files, key=os.path.getmtime)

def find_latest_hdf5(base_dir='/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7'):
    files = glob.glob(os.path.join(base_dir, '*/*.hdf5'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ──────────────────────────────────────────────
# 메인 뷰어
# ──────────────────────────────────────────────
def main(filepath=None):
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
    demos = sorted(data_group.keys())
    if not demos:
        print("Error: No demos found.")
        f.close()
        return

    print(f"[INFO] Found {len(demos)} demo(s): {list(demos)}")

    # ── 현재 상태 ──────────────────────────────
    state = {
        'demo_idx': 0,
        'frame': 0,
        'num_frames': 1,
        'obs': None,
        'img_keys': [],
        'dirty': True,      # 그래프 재그리기 필요 여부
        'ref_num_frames': 1, # 20Hz reference (images/joints)
    }

    def load_demo(idx):
        state['demo_idx'] = idx
        obs = data_group[demos[idx]]['observations']
        state['obs'] = obs

        # 이미지 키
        state['img_keys'] = [k for k in ['image_H', 'image_T', 'image_L', 'image_R'] if k in obs]

        # 프레임 수 (로봇 데이터 기준 / 없으면 아무 키나)
        for k in ['joint_L', 'joint_R', 'image_H', 'image_T', 'image_L', 'image_R']:
            if k in obs and len(obs[k]) > 0:
                state['num_frames'] = len(obs[k])
                break
        else:
            state['num_frames'] = 1

        state['frame'] = 0
        state['dirty'] = True

        print(f"\n[INFO] Demo {demos[idx]}: {state['num_frames']} frames")
        print(f"       Keys: {list(obs.keys())}")

        # OpenCV 트랙바 최대값 갱신
        cv2.setTrackbarMax('Frame', 'Viewer', max(state['num_frames'] - 1, 1))
        cv2.setTrackbarPos('Frame', 'Viewer', 0)

    # ── OpenCV 뷰어 ───────────────────────────
    cv2.namedWindow('Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Viewer', 1280, 480)

    # 트랙바: Frame
    def on_frame_trackbar(val):
        state['frame'] = val

    def on_demo_trackbar(val):
        load_demo(val)

    cv2.createTrackbar('Frame', 'Viewer', 0, 1, on_frame_trackbar)
    cv2.createTrackbar('Demo',  'Viewer', 0, max(len(demos) - 1, 1), on_demo_trackbar)

    # ── Matplotlib 그래프 (별도 스레드) ────────
    fig_data = {'fig': None, 'lines_joint': [], 'lines_wrench': [],
                 'marker_j': None, 'marker_w': None}

    def build_graph():
        obs = state['obs']
        if obs is None:
            return

        if fig_data['fig'] is not None:
            plt.close(fig_data['fig'])

        fig = plt.figure(figsize=(14, 6))
        fig.canvas.manager.set_window_title(f"Graph – {demos[state['demo_idx']]}")
        gs = gridspec.GridSpec(1, 2, figure=fig)
        ax_j = fig.add_subplot(gs[0, 0])
        ax_w = fig.add_subplot(gs[0, 1])

        # Joint 그래프
        joint_keys = [k for k in ['joint_L', 'joint_R'] if k in obs]
        for jk in joint_keys:
            arr = np.array(obs[jk])
            if arr.ndim == 2:
                for i in range(arr.shape[1]):
                    ax_j.plot(arr[:, i], label=f'{jk[-1]} J{i+1}')
        ax_j.set_title("Joint Angles")
        if joint_keys:
            ax_j.legend(loc='upper right', fontsize=7, ncol=2)

        # F_e_raw 그래프 (기존 Wrench 대신)
        if 'F_e_raw' in obs:
            arr = np.array(obs['F_e_raw'])
            if arr.ndim == 2:
                # 처음 3개는 Force (Fx, Fy, Fz)
                for i, lb in enumerate(['Fx','Fy','Fz']):
                    if i < arr.shape[1]:
                        ax_w.plot(arr[:, i], label=lb)
        ax_w.set_title("External Force (F_e_raw)")
        if 'F_e_raw' in obs:
            ax_w.legend(loc='upper right', fontsize=7, ncol=3)

        # 프레임 마커
        mk_j = ax_j.axvline(0, color='red', lw=1.2, ls='--', label='_nolegend_')
        mk_w = ax_w.axvline(0, color='red', lw=1.2, ls='--', label='_nolegend_')

        fig_data['fig'] = fig
        fig_data['ax_j'] = ax_j
        fig_data['ax_w'] = ax_w
        fig_data['mk_j'] = mk_j
        fig_data['mk_w'] = mk_w

        plt.tight_layout()
        plt.show(block=False)

    # ── 초기 로드 ─────────────────────────────
    load_demo(0)
    build_graph()

    prev_frame = -1
    prev_demo  = -1

    # ── 메인 루프 ─────────────────────────────
    while True:
        demo_idx = cv2.getTrackbarPos('Demo',  'Viewer')
        frame    = cv2.getTrackbarPos('Frame', 'Viewer')

        # 데모 전환
        if demo_idx != prev_demo:
            prev_demo = demo_idx
            load_demo(demo_idx)
            build_graph()
            prev_frame = -1     # 이미지 강제 갱신

        obs = state['obs']
        if obs is None:
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        frame = min(frame, state['num_frames'] - 1)

        # 이미지 갱신
        if frame != prev_frame:
            prev_frame = frame

            imgs = []
            for k in state['img_keys']:
                if frame < len(obs[k]):
                    img = obs[k][frame]
                    if img is not None and img.size > 0:
                        imgs.append(img)  # BGR 그대로 OpenCV에 표시

            if imgs:
                # 높이 맞춰 hstack
                h = imgs[0].shape[0]
                resized = []
                for im in imgs:
                    if im.shape[0] != h:
                        im = cv2.resize(im, (int(im.shape[1]*h/im.shape[0]), h))
                    resized.append(im)
                combined = np.hstack(resized)
                cv2.putText(combined, f"Demo: {demos[demo_idx]}  Frame: {frame}/{state['num_frames']-1}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100), 2)
                cv2.imshow('Viewer', combined)
            else:
                blank = np.zeros((240, 640, 3), np.uint8)
                cv2.putText(blank, "No image data", (20,120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
                cv2.imshow('Viewer', blank)

            # Matplotlib 마커 업데이트 (데이터 주기에 맞춰 스케일링)
            if fig_data.get('mk_j') is not None:
                try:
                    # Joint 마커 (보통 20Hz)
                    ratio_j = 1.0
                    if 'joint_L' in obs:
                        ratio_j = len(obs['joint_L']) / state['num_frames']
                    fig_data['mk_j'].set_xdata([frame * ratio_j, frame * ratio_j])
                    
                    # F_e_raw 마커 (보통 250Hz)
                    ratio_w = 1.0
                    if 'F_e_raw' in obs:
                        ratio_w = len(obs['F_e_raw']) / state['num_frames']
                    fig_data['mk_w'].set_xdata([frame * ratio_w, frame * ratio_w])
                    
                    fig_data['fig'].canvas.draw_idle()
                    fig_data['fig'].canvas.flush_events()
                except Exception:
                    pass

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):   # 다음 데모
            new_idx = min(demo_idx + 1, len(demos) - 1)
            cv2.setTrackbarPos('Demo', 'Viewer', new_idx)
        elif key == ord('p'):   # 이전 데모
            new_idx = max(demo_idx - 1, 0)
            cv2.setTrackbarPos('Demo', 'Viewer', new_idx)

    cv2.destroyAllWindows()
    if fig_data.get('fig') is not None:
        plt.close(fig_data['fig'])
    f.close()
    print("\n[INFO] Viewer closed.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None,
                        help="HDF5 file path. If omitted, uses the latest file.")
    args = parser.parse_args()
    main(args.file)
