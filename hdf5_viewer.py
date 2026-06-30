import sys
import h5py
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox, 
                             QScrollArea, QSlider, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 Data Viewer")
        self.resize(1200, 800)
        
        self.hdf5_file = None
        self.current_demo = None
        self.data_cache = {} # demo_name -> dict of arrays
        self.image_keys = []
        self.plot_keys = []
        self.checkboxes = {}
        self.checked_keys_ax1 = set()
        self.checked_keys_ax2 = set()

        # 재생 타이머 (20Hz 기본)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._is_playing = False
        self._play_hz = 20.0  # 기본 재생 속도
        
        self.initUI()
        
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # --- Top control bar ---
        top_bar = QHBoxLayout()
        self.btn_open = QPushButton("Open HDF5 File")
        self.btn_open.clicked.connect(self.open_file)
        self.lbl_file = QLabel("No file selected")
        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.lbl_file)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        # --- Demo selection bar ---
        demo_bar = QHBoxLayout()
        demo_bar.addWidget(QLabel("Select Demo:"))
        self.combo_demo = QComboBox()
        self.combo_demo.currentIndexChanged.connect(self.on_demo_changed)
        demo_bar.addWidget(self.combo_demo)
        
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(self.prev_demo)
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_demo)
        demo_bar.addWidget(self.btn_prev)
        demo_bar.addWidget(self.btn_next)
        
        # Allow typing a demo name directly
        self.combo_demo.setEditable(True)
        self.combo_demo.lineEdit().returnPressed.connect(self.on_demo_text_entered)

        demo_bar.addStretch()
        main_layout.addLayout(demo_bar)
        
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
        
        # Plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212, sharex=self.ax1)
        right_layout.addWidget(self.canvas, stretch=2)
        
        # Image and Slider
        img_slider_layout = QHBoxLayout()
        self.lbl_image = QLabel("No Image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(320, 240)
        img_slider_layout.addWidget(self.lbl_image, stretch=1)
        
        right_layout.addLayout(img_slider_layout, stretch=1)
        
        # Slider + Play controls
        slider_layout = QHBoxLayout()

        # 재생 버튼 ▶/⏸
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(36)
        self.btn_play.setToolTip("Play / Pause (20 Hz)")
        self.btn_play.clicked.connect(self.toggle_play)

        # 정지 버튼 ⏹
        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedWidth(36)
        self.btn_stop.setToolTip("Stop and rewind")
        self.btn_stop.clicked.connect(self.stop_play)

        # 속도 배율
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.combo_speed.setCurrentIndex(2)  # 1× 기본
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

    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open HDF5 File", "", "HDF5 Files (*.hdf5 *.h5);;All Files (*)", options=options)
        if file_name:
            self.load_file(file_name)
            
    def load_file(self, file_path):
        if self.hdf5_file:
            self.hdf5_file.close()
            
        try:
            self.hdf5_file = h5py.File(file_path, 'r')
            self.lbl_file.setText(file_path)
            
            if 'data' not in self.hdf5_file:
                print("Error: 'data' group not found in HDF5 file.")
                return
                
            demo_keys = list(self.hdf5_file['data'].keys())
            # Sort demo keys numerically (e.g. demo_0, demo_1, demo_10)
            demo_keys.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
            
            self.combo_demo.blockSignals(True)
            self.combo_demo.clear()
            self.combo_demo.addItems(demo_keys)
            self.combo_demo.blockSignals(False)
            
            if demo_keys:
                self.combo_demo.setCurrentIndex(0)
                self.on_demo_changed()
        except Exception as e:
            print(f"Failed to open file: {e}")

    def on_demo_text_entered(self):
        text = self.combo_demo.currentText()
        idx = self.combo_demo.findText(text)
        if idx != -1:
            self.combo_demo.setCurrentIndex(idx)
        else:
            # If they just entered a number like "5", try to find "demo_5"
            if text.isdigit():
                demo_name = f"demo_{text}"
                idx = self.combo_demo.findText(demo_name)
                if idx != -1:
                    self.combo_demo.setCurrentIndex(idx)
                    return
            print(f"Demo '{text}' not found.")

    def prev_demo(self):
        idx = self.combo_demo.currentIndex()
        if idx > 0:
            self.combo_demo.setCurrentIndex(idx - 1)
            
    def next_demo(self):
        idx = self.combo_demo.currentIndex()
        if idx < self.combo_demo.count() - 1:
            self.combo_demo.setCurrentIndex(idx + 1)

    def extract_datasets(self, group, prefix=''):
        items = {}
        for key, item in group.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                items[path] = item[:]
            elif isinstance(item, h5py.Group):
                items.update(self.extract_datasets(item, path))
        return items

    def on_demo_changed(self):
        demo_name = self.combo_demo.currentText()
        if not demo_name or not self.hdf5_file:
            return
            
        self.current_demo = demo_name
        
        # Load data for this demo
        if demo_name not in self.hdf5_file['data']:
            print(f"Demo '{demo_name}' not found in file.")
            return
            
        demo_group = self.hdf5_file['data'][demo_name]
        data = self.extract_datasets(demo_group)
        
        self.image_keys = []
        self.plot_keys = []
        self.data_cache = {}
        
        max_len = 0
        
        for k, v in data.items():
            try:
                if len(v.shape) == 0:
                    continue
                max_len = max(max_len, v.shape[0])
                self.data_cache[k] = v
                if len(v.shape) >= 3 and v.dtype == np.uint8:
                    self.image_keys.append(k)
                elif len(v.shape) >= 3 and v.shape[-1] in [1, 3]:
                    self.image_keys.append(k)
                else:
                    self.plot_keys.append(k)
            except Exception:
                pass

        # ── 길이 정렬: joint_L 기준으로 더 긴 배열 앞부분 제거 ──────
        # 방법 1: timestamp_robot / timestamp_wrench가 있으면 nearest-neighbor
        ts_r_key = next((k for k in self.data_cache if 'timestamp_robot' in k), None)
        ts_w_key = next((k for k in self.data_cache if 'timestamp_wrench' in k), None)

        if ts_r_key and ts_w_key:
            ts_r = self.data_cache[ts_r_key]   # (N_r,)
            ts_w = self.data_cache[ts_w_key]   # (N_w,)
            N_r, N_w = len(ts_r), len(ts_w)
            if N_r != N_w:
                align_idx = np.array(
                    [np.argmin(np.abs(ts_w - t)) for t in ts_r]
                )
                for k in list(self.data_cache.keys()):
                    v = self.data_cache[k]
                    if len(v.shape) > 0 and v.shape[0] == N_w and k != ts_r_key:
                        self.data_cache[k] = v[align_idx]
                max_len = N_r
                print(f"[align] {demo_name}: N_w={N_w} → N_r={N_r} (nearest-neighbor)")
        else:
            # 방법 2: timestamp 없으면 min-length로 앞부분 잘라냄
            lengths = [v.shape[0] for v in self.data_cache.values() if len(v.shape) > 0]
            if lengths:
                min_len = min(lengths)
                if min_len != max(lengths):
                    for k in list(self.data_cache.keys()):
                        v = self.data_cache[k]
                        if len(v.shape) > 0 and v.shape[0] > min_len:
                            self.data_cache[k] = v[-min_len:]
                    max_len = min_len
                    print(f"[align] {demo_name}: min-length truncate → {min_len}")
        # ──────────────────────────────────────────────────────────────
                
        # Setup slider
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

    def _add_checkbox_row(self, key):
        row = QHBoxLayout()
        cb1 = QCheckBox("1")
        cb2 = QCheckBox("2")
        cb1.setToolTip("Plot on Subplot 1")
        cb2.setToolTip("Plot on Subplot 2")
        
        cb1.setChecked(key in self.checked_keys_ax1)
        cb2.setChecked(key in self.checked_keys_ax2)
        
        cb1.stateChanged.connect(self.update_plot)
        cb2.stateChanged.connect(self.update_plot)
        
        row.addWidget(cb1)
        row.addWidget(cb2)
        row.addWidget(QLabel(key))
        row.addStretch()
        
        container = QWidget()
        container.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        self.layout_plots.addWidget(container)
        
        self.checkboxes[key] = {'ax1': cb1, 'ax2': cb2}

    def populate_controls(self):
        # 현재 체크 상태 저장
        for key, cbs in self.checkboxes.items():
            if cbs['ax1'].isChecked():
                self.checked_keys_ax1.add(key)
            else:
                self.checked_keys_ax1.discard(key)
            if cbs['ax2'].isChecked():
                self.checked_keys_ax2.add(key)
            else:
                self.checked_keys_ax2.discard(key)

        # Clear old checkboxes
        for i in reversed(range(self.layout_plots.count())):
            item = self.layout_plots.itemAt(i)
            widgetToRemove = item.widget()
            self.layout_plots.removeItem(item)
            if widgetToRemove is not None:
                widgetToRemove.setParent(None)

        self.checkboxes = {}
        for key in sorted(self.plot_keys):
            data = self.data_cache[key]
            if len(data.shape) == 1:
                self._add_checkbox_row(key)
            elif len(data.shape) == 2:
                # Create a label for grouping
                lbl = QLabel(f"<b>{key}</b>")
                self.layout_plots.addWidget(lbl)
                for i in range(data.shape[1]):
                    sub_key = f"{key}[{i}]"
                    self._add_checkbox_row(sub_key)

        # Add stretch to layout so checkboxes are at top
        self.layout_plots.addStretch()

        # Clear old image combo
        self.combo_img.blockSignals(True)
        self.combo_img.clear()
        self.combo_img.addItems(self.image_keys)
        self.combo_img.blockSignals(False)

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.vline1 = None
        self.vline2 = None
        
        plotted_any1 = False
        plotted_any2 = False
        
        for key_idx, cbs in self.checkboxes.items():
            cb1, cb2 = cbs['ax1'], cbs['ax2']
            if cb1.isChecked() or cb2.isChecked():
                # Extract original key and index
                if '[' in key_idx and key_idx.endswith(']'):
                    base_key = key_idx[:key_idx.index('[')]
                    idx = int(key_idx[key_idx.index('[')+1:-1])
                    data = self.data_cache[base_key][:, idx]
                else:
                    base_key = key_idx
                    data = self.data_cache[base_key]
                
                if cb1.isChecked():
                    self.ax1.plot(data, label=key_idx)
                    plotted_any1 = True
                if cb2.isChecked():
                    self.ax2.plot(data, label=key_idx)
                    plotted_any2 = True
                    
        if plotted_any1:
            self.ax1.legend(loc='upper right', fontsize='small')
            self.ax1.grid(True)
            step = self.slider.value()
            self.vline1 = self.ax1.axvline(x=step, color='r', linestyle='--', linewidth=1.5)
            
        if plotted_any2:
            self.ax2.legend(loc='upper right', fontsize='small')
            self.ax2.grid(True)
            step = self.slider.value()
            self.vline2 = self.ax2.axvline(x=step, color='r', linestyle='--', linewidth=1.5)
            
        self.figure.tight_layout()
        self.canvas.draw()

    def on_slider_changed(self):
        step = self.slider.value()
        max_step = self.slider.maximum()
        self.lbl_step.setText(f"Step: {step} / {max_step}")
        
        needs_draw = False
        if self.vline1 is not None:
            self.vline1.set_xdata([step, step])
            needs_draw = True
        if self.vline2 is not None:
            self.vline2.set_xdata([step, step])
            needs_draw = True
            
        if needs_draw:
            self.canvas.draw_idle()
            
        self.update_image()

    def update_image(self):
        img_key = self.combo_img.currentText()
        if not img_key or img_key not in self.data_cache:
            self.lbl_image.setText("No Image")
            return
            
        step = self.slider.value()
        img_data = self.data_cache[img_key]
        
        if step >= img_data.shape[0]:
            step = img_data.shape[0] - 1
            
        img_frame = img_data[step]
        
        # Convert to QImage
        # Assume shape is (H, W, 3) and uint8 or float
        if img_frame.dtype != np.uint8:
            if img_frame.max() <= 1.0:
                img_frame = (img_frame * 255).astype(np.uint8)
            else:
                img_frame = img_frame.astype(np.uint8)
                
        # If shape is (3, H, W), transpose to (H, W, 3)
        if len(img_frame.shape) == 3 and img_frame.shape[0] == 3:
            img_frame = np.transpose(img_frame, (1, 2, 0))
            
        if len(img_frame.shape) == 3 and img_frame.shape[2] == 3:
            h, w, c = img_frame.shape
            bytes_per_line = c * w
            # Important: Ensure contiguous memory for QImage
            img_frame = np.ascontiguousarray(img_frame)
            qimg = QImage(img_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl_image.setPixmap(pixmap.scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_image.setText(f"Unsupported image shape: {img_frame.shape}")

    # ── 재생 제어 ────────────────────────────────────────────────
    def toggle_play(self):
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if not self.slider.isEnabled():
            return
        # 끝에 있으면 처음부터
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self._is_playing = True
        self.btn_play.setText("⏸")
        interval_ms = int(1000.0 / self._play_hz)
        self._play_timer.start(interval_ms)

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
            self._pause()   # 끝에 도달 → 자동 정지
            return
        self.slider.setValue(cur + 1)

    def _on_speed_changed(self):
        speed_map = {"0.25×": 0.25, "0.5×": 0.5, "1×": 1.0, "2×": 2.0, "4×": 4.0}
        label = self.combo_speed.currentText()
        self._play_hz = 20.0 * speed_map.get(label, 1.0)
        if self._is_playing:
            # 속도 변경을 즉시 반영
            self._play_timer.setInterval(int(1000.0 / self._play_hz))
    # ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    
    # Optional: Load default file if passed as argument
    if len(sys.argv) > 1:
        viewer.load_file(sys.argv[1])
        
    viewer.show()
    sys.exit(app.exec_())

