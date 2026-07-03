"""
zarr_viewer.py  —  raw zarr 에피소드 데이터셋 뷰어

hdf5_viewer.py 와 동일한 UX 로, 학습 이전의 raw zarr 데이터를 직접 열어본다.

지원하는 폴더 구조 (rush 데이터 수집 포맷):
    dataset_root/
        episode_000/
            robot/ ee_pose_se3.zarr (N,4,4), command_pose_se3.zarr, joint_deg.zarr ...
            ft/    wrench_raw.zarr, jt_tared_wrench.zarr ...
            camera_0_D405/ rgb.zarr (N,480,640,3), depth.zarr (N,480,640) ...
            camera_1_D435/ ...
        episode_001/ ...

- "Open Dataset Folder" 로 dataset_root(= episode_* 들을 담은 폴더) 를 열면
  에피소드 콤보박스가 채워진다. episode_* 가 없는 단일 에피소드 폴더를 직접 열어도 된다.
- 좌측: 플롯할 변수 체크박스(서브플롯 1/2). (N,4,4) 같은 다차원 배열은 성분별로 펼쳐서 선택.
- 우측: 선택한 카메라 이미지(rgb/depth)를 슬라이더 시점에 맞춰 표시, ▶ 재생 지원.
- 이미지/depth 는 통째로 메모리에 올리지 않고 프레임 단위로 lazy 인덱싱한다.

Usage:
    python zarr_viewer.py [dataset_root]
"""

import sys
import os

import numpy as np
import zarr

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox,
                             QScrollArea, QSlider, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm


def find_zarr_arrays(root_dir):
    """root_dir 하위의 모든 *.zarr 배열을 찾아 {상대키: 절대경로} 로 반환."""
    found = {}
    for dirpath, dirnames, _ in os.walk(root_dir):
        # *.zarr 디렉토리를 만나면 그 자체가 배열이므로 하위 탐색 중단
        keep = []
        for d in dirnames:
            full = os.path.join(dirpath, d)
            if d.endswith('.zarr'):
                rel = os.path.relpath(full, root_dir)
                key = rel[:-len('.zarr')]           # 뒤의 .zarr 제거
                found[key.replace(os.sep, '/')] = full
            else:
                keep.append(d)
        dirnames[:] = keep
    return found


def classify_array(shape, dtype):
    """배열 shape/dtype 으로 'rgb' / 'gray'(depth) / 'plot' 분류."""
    if len(shape) >= 3:
        # (..., H, W, 3) RGB
        if shape[-1] == 3 and shape[-2] >= 8:
            return 'rgb'
        # (N, H, W) 단일채널(depth 등)
        if len(shape) == 3 and shape[-1] >= 8 and shape[-2] >= 8:
            return 'gray'
    return 'plot'


class ZarrViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zarr Data Viewer")
        self.resize(1200, 800)

        self.dataset_root = None
        self.episode_dirs = {}          # episode name -> 절대경로
        self.current_episode = None

        self.plot_data = {}             # key -> np.ndarray (작은 배열, 전량 로드)
        self.image_arrays = {}          # key -> zarr array (lazy 인덱싱)
        self.image_keys = []
        self.plot_keys = []

        self.series = {}                # label -> (base_key, flat_col or None)
        self.checkboxes = {}            # label -> {'ax1':cb, 'ax2':cb}
        self.checked_keys_ax1 = set()
        self.checked_keys_ax2 = set()

        # 재생 타이머
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._is_playing = False
        self._play_hz = 20.0

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Top control bar ---
        top_bar = QHBoxLayout()
        self.btn_open = QPushButton("Open Dataset Folder")
        self.btn_open.clicked.connect(self.open_folder)
        self.lbl_file = QLabel("No folder selected")
        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.lbl_file)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # --- Episode selection bar ---
        ep_bar = QHBoxLayout()
        ep_bar.addWidget(QLabel("Select Episode:"))
        self.combo_ep = QComboBox()
        self.combo_ep.currentIndexChanged.connect(self.on_episode_changed)
        ep_bar.addWidget(self.combo_ep)

        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(self.prev_episode)
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_episode)
        ep_bar.addWidget(self.btn_prev)
        ep_bar.addWidget(self.btn_next)
        ep_bar.addStretch()
        main_layout.addLayout(ep_bar)

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Left Panel (Checkboxes)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.group_plots = QGroupBox("Variables to Plot")
        self.layout_plots = QVBoxLayout()
        scroll_plots = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.layout_plots)
        scroll_plots.setWidget(scroll_widget)
        scroll_plots.setWidgetResizable(True)

        layout_group = QVBoxLayout()
        layout_group.addWidget(scroll_plots)
        self.group_plots.setLayout(layout_group)
        left_layout.addWidget(self.group_plots)

        # Image Selection
        self.group_img = QGroupBox("Image to Display")
        img_layout = QVBoxLayout()
        self.combo_img = QComboBox()
        self.combo_img.currentIndexChanged.connect(self.update_image)
        img_layout.addWidget(self.combo_img)
        self.group_img.setLayout(img_layout)
        left_layout.addWidget(self.group_img)

        splitter.addWidget(left_panel)

        # Right Panel (Visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212, sharex=self.ax1)
        right_layout.addWidget(self.canvas, stretch=2)

        img_slider_layout = QHBoxLayout()
        self.lbl_image = QLabel("No Image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(320, 240)
        img_slider_layout.addWidget(self.lbl_image, stretch=1)
        right_layout.addLayout(img_slider_layout, stretch=1)

        # Slider + Play controls
        slider_layout = QHBoxLayout()

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(36)
        self.btn_play.setToolTip("Play / Pause (20 Hz)")
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedWidth(36)
        self.btn_stop.setToolTip("Stop and rewind")
        self.btn_stop.clicked.connect(self.stop_play)

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.combo_speed.setCurrentIndex(2)
        self.combo_speed.setFixedWidth(60)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.lbl_step = QLabel("Step: 0 / 0")

        slider_layout.addWidget(self.btn_play)
        slider_layout.addWidget(self.btn_stop)
        slider_layout.addWidget(QLabel("Speed:"))
        slider_layout.addWidget(self.combo_speed)
        slider_layout.addWidget(QLabel("  Step:"))
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.lbl_step)
        right_layout.addLayout(slider_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        self.vline1 = None
        self.vline2 = None

    # ── 폴더/에피소드 로드 ───────────────────────────────────────
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Dataset Folder", "")
        if folder:
            self.load_dataset(folder)

    def load_dataset(self, folder):
        self.dataset_root = folder
        self.lbl_file.setText(folder)

        # episode_* 하위 폴더 탐색
        eps = sorted([
            d for d in os.listdir(folder)
            if d.startswith("episode_") and os.path.isdir(os.path.join(folder, d))
        ])

        if eps:
            self.episode_dirs = {e: os.path.join(folder, e) for e in eps}
        else:
            # 단일 에피소드 폴더를 직접 연 경우
            name = os.path.basename(os.path.normpath(folder))
            self.episode_dirs = {name: folder}

        ep_keys = sorted(self.episode_dirs.keys())
        self.combo_ep.blockSignals(True)
        self.combo_ep.clear()
        self.combo_ep.addItems(ep_keys)
        self.combo_ep.blockSignals(False)

        if ep_keys:
            self.combo_ep.setCurrentIndex(0)
            self.on_episode_changed()

    def prev_episode(self):
        idx = self.combo_ep.currentIndex()
        if idx > 0:
            self.combo_ep.setCurrentIndex(idx - 1)

    def next_episode(self):
        idx = self.combo_ep.currentIndex()
        if idx < self.combo_ep.count() - 1:
            self.combo_ep.setCurrentIndex(idx + 1)

    def on_episode_changed(self):
        ep_name = self.combo_ep.currentText()
        if not ep_name or ep_name not in self.episode_dirs:
            return
        self.current_episode = ep_name
        ep_path = self.episode_dirs[ep_name]

        arrays = find_zarr_arrays(ep_path)

        self.plot_data = {}
        self.image_arrays = {}
        self.image_keys = []
        self.plot_keys = []
        max_len = 0

        for key, path in sorted(arrays.items()):
            try:
                z = zarr.open(path, mode='r')
                shape = z.shape
                if len(shape) == 0:
                    continue
                max_len = max(max_len, shape[0])
                kind = classify_array(shape, z.dtype)
                if kind in ('rgb', 'gray'):
                    self.image_arrays[key] = z            # lazy
                    self.image_keys.append(key)
                else:
                    self.plot_data[key] = z[:]            # 전량 로드(작음)
                    self.plot_keys.append(key)
            except Exception as e:
                print(f"[skip] {key}: {e}")

        # Slider 설정
        if max_len > 0:
            self.slider.setEnabled(True)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max_len - 1)
            self.slider.setValue(0)
            self.lbl_step.setText(f"Step: 0 / {max_len - 1}")
        else:
            self.slider.setEnabled(False)

        self.populate_controls()
        self.update_plot()
        self.update_image()

    # ── 체크박스 UI ──────────────────────────────────────────────
    def _add_checkbox_row(self, label):
        row = QHBoxLayout()
        cb1 = QCheckBox("1")
        cb2 = QCheckBox("2")
        cb1.setToolTip("Plot on Subplot 1")
        cb2.setToolTip("Plot on Subplot 2")
        cb1.setChecked(label in self.checked_keys_ax1)
        cb2.setChecked(label in self.checked_keys_ax2)
        cb1.stateChanged.connect(self.update_plot)
        cb2.stateChanged.connect(self.update_plot)
        row.addWidget(cb1)
        row.addWidget(cb2)
        row.addWidget(QLabel(label))
        row.addStretch()
        container = QWidget()
        container.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        self.layout_plots.addWidget(container)
        self.checkboxes[label] = {'ax1': cb1, 'ax2': cb2}

    def populate_controls(self):
        # 현재 체크 상태 저장
        for label, cbs in self.checkboxes.items():
            (self.checked_keys_ax1.add if cbs['ax1'].isChecked() else self.checked_keys_ax1.discard)(label)
            (self.checked_keys_ax2.add if cbs['ax2'].isChecked() else self.checked_keys_ax2.discard)(label)

        # 기존 체크박스 제거
        for i in reversed(range(self.layout_plots.count())):
            item = self.layout_plots.itemAt(i)
            w = item.widget()
            self.layout_plots.removeItem(item)
            if w is not None:
                w.setParent(None)

        self.checkboxes = {}
        self.series = {}

        for key in sorted(self.plot_keys):
            data = self.plot_data[key]
            if data.ndim == 1:
                self.series[key] = (key, None)
                self._add_checkbox_row(key)
            else:
                # 다차원 → (N, -1) 성분별 라벨
                trailing = data.shape[1:]
                n_cols = int(np.prod(trailing))
                lbl = QLabel(f"<b>{key}</b>")
                self.layout_plots.addWidget(lbl)
                for c in range(n_cols):
                    idx = np.unravel_index(c, trailing)
                    idx_str = ",".join(str(i) for i in idx)
                    label = f"{key}[{idx_str}]"
                    self.series[label] = (key, c)
                    self._add_checkbox_row(label)

        self.layout_plots.addStretch()

        # 이미지 콤보 갱신
        self.combo_img.blockSignals(True)
        self.combo_img.clear()
        self.combo_img.addItems(self.image_keys)
        self.combo_img.blockSignals(False)

    # ── 플롯 ─────────────────────────────────────────────────────
    def _series_values(self, label):
        base_key, col = self.series[label]
        data = self.plot_data[base_key]
        if col is None:
            return data
        return data.reshape(data.shape[0], -1)[:, col]

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.vline1 = None
        self.vline2 = None

        plotted1 = plotted2 = False
        for label, cbs in self.checkboxes.items():
            cb1, cb2 = cbs['ax1'], cbs['ax2']
            if not (cb1.isChecked() or cb2.isChecked()):
                continue
            values = self._series_values(label)
            if cb1.isChecked():
                self.ax1.plot(values, label=label)
                plotted1 = True
            if cb2.isChecked():
                self.ax2.plot(values, label=label)
                plotted2 = True

        step = self.slider.value()
        if plotted1:
            self.ax1.legend(loc='upper right', fontsize='small')
            self.ax1.grid(True)
            self.vline1 = self.ax1.axvline(x=step, color='r', linestyle='--', linewidth=1.5)
        if plotted2:
            self.ax2.legend(loc='upper right', fontsize='small')
            self.ax2.grid(True)
            self.vline2 = self.ax2.axvline(x=step, color='r', linestyle='--', linewidth=1.5)

        self.figure.tight_layout()
        self.canvas.draw()

    def on_slider_changed(self):
        step = self.slider.value()
        self.lbl_step.setText(f"Step: {step} / {self.slider.maximum()}")
        needs_draw = False
        if self.vline1 is not None:
            self.vline1.set_xdata([step, step]); needs_draw = True
        if self.vline2 is not None:
            self.vline2.set_xdata([step, step]); needs_draw = True
        if needs_draw:
            self.canvas.draw_idle()
        self.update_image()

    # ── 이미지 (lazy) ────────────────────────────────────────────
    def update_image(self):
        img_key = self.combo_img.currentText()
        if not img_key or img_key not in self.image_arrays:
            self.lbl_image.setText("No Image")
            return

        z = self.image_arrays[img_key]
        step = min(self.slider.value(), z.shape[0] - 1)
        frame = np.asarray(z[step])

        # (3,H,W) → (H,W,3)
        if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))

        if frame.ndim == 2:
            # depth/gray → 정규화 후 turbo 컬러맵
            f = frame.astype(np.float32)
            valid = f[f > 0]
            vmin = valid.min() if valid.size else 0.0
            vmax = f.max() if f.max() > vmin else vmin + 1.0
            norm = np.clip((f - vmin) / (vmax - vmin), 0, 1)
            rgb = (cm.turbo(norm)[:, :, :3] * 255).astype(np.uint8)
            frame = rgb
        elif frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        if frame.ndim == 3 and frame.shape[2] == 3:
            h, w, c = frame.shape
            frame = np.ascontiguousarray(frame)
            qimg = QImage(frame.data, w, h, c * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl_image.setPixmap(
                pixmap.scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_image.setText(f"Unsupported image shape: {frame.shape}")

    # ── 재생 제어 ────────────────────────────────────────────────
    def toggle_play(self):
        self._pause() if self._is_playing else self._play()

    def _play(self):
        if not self.slider.isEnabled():
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self._is_playing = True
        self.btn_play.setText("⏸")
        self._play_timer.start(int(1000.0 / self._play_hz))

    def _pause(self):
        self._is_playing = False
        self._play_timer.stop()
        self.btn_play.setText("▶")

    def stop_play(self):
        self._pause()
        self.slider.setValue(0)

    def _play_tick(self):
        cur = self.slider.value()
        if cur >= self.slider.maximum():
            self._pause()
            return
        self.slider.setValue(cur + 1)

    def _on_speed_changed(self):
        speed_map = {"0.25×": 0.25, "0.5×": 0.5, "1×": 1.0, "2×": 2.0, "4×": 4.0}
        self._play_hz = 20.0 * speed_map.get(self.combo_speed.currentText(), 1.0)
        if self._is_playing:
            self._play_timer.setInterval(int(1000.0 / self._play_hz))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ZarrViewer()
    if len(sys.argv) > 1:
        viewer.load_dataset(sys.argv[1])
    viewer.show()
    sys.exit(app.exec_())
