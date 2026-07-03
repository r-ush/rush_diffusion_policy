#!/usr/bin/env python3
"""
HDF5 Viewer — Left Hand FK + Zeroset Wrench
- 3D hand kinematic skeleton with force arrows (zeroset data)
- Camera images (fixed size, no resize on slider move)
- Single wrench plot: per-finger checkbox to select which fingers to show
- Multi-demo overlay via demo checkboxes
"""

import sys, os
import numpy as np
import h5py
import xml.etree.ElementTree as ET

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QPushButton, QCheckBox, QGroupBox, QScrollArea,
    QComboBox, QSplitter, QFileDialog, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

# ── Constants ─────────────────────────────────────────────────────────────────
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "(reference)aidin_hand_description/urdf/hand.urdf")
HDF5_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "common_data.hdf5")

FINGERS = ['thumb', 'index', 'middle', 'ring', 'baby']
FCOLORS = {
    'thumb':  '#e74c3c',
    'index':  '#3498db',
    'middle': '#2ecc71',
    'ring':   '#f39c12',
    'baby':   '#9b59b6',
}

# ── FK helpers ────────────────────────────────────────────────────────────────

def make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3, 3]  = xyz
    return T

def rot_axis_angle(axis, angle):
    ax = np.array(axis, float)
    ax /= np.linalg.norm(ax) + 1e-9
    R = np.eye(4)
    R[:3, :3] = Rotation.from_rotvec(ax * angle).as_matrix()
    return R

class URDFHand:
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        for j in root.findall('joint'):
            name = j.get('name', '')
            if 'left' not in name:
                continue
            orig = j.find('origin')
            xyz = list(map(float, orig.get('xyz','0 0 0').split())) if orig is not None else [0,0,0]
            rpy = list(map(float, orig.get('rpy','0 0 0').split())) if orig is not None else [0,0,0]
            ax_el = j.find('axis')
            axis = list(map(float, ax_el.get('xyz','0 0 1').split())) if ax_el is not None else [0,0,1]
            p = j.find('parent'); c = j.find('child')
            self.joints[name] = {
                'type':   j.get('type'),
                'xyz':    np.array(xyz),
                'rpy':    np.array(rpy),
                'axis':   np.array(axis),
                'parent': p.get('link') if p is not None else '',
                'child':  c.get('link')  if c is not None else '',
            }

    def fk_finger(self, finger, q3):
        """q3: [j1, j2, j3]; j4 mirrors j3. Returns list of 4x4 transforms."""
        vals = list(q3) + [q3[-1]]
        names = [f'left_{finger}_joint{i}' for i in range(1, 5)]
        T = np.eye(4)
        out = []
        for i, jn in enumerate(names):
            if jn not in self.joints:
                break
            j = self.joints[jn]
            T = T @ make_T(j['xyz'], j['rpy'])
            if j['type'] == 'revolute':
                T = T @ rot_axis_angle(j['axis'], vals[i])
            out.append(T.copy())
        return out

    def all_fk(self, hand_q15):
        return {f: self.fk_finger(f, hand_q15[i*3:(i+1)*3]) for i, f in enumerate(FINGERS)}

# ── Data ──────────────────────────────────────────────────────────────────────

class DemoData:
    def __init__(self, f, key):
        obs = f[f'data/{key}/observations']
        self.hand_L   = obs['hand_L'][:]
        self.image_H  = obs['image_H'][:]
        self.image_T  = obs['image_T'][:]
        self.ts_robot  = obs['timestamp_robot'][:]
        self.ts_wrench = obs['timestamp_wrench'][:]
        self.n_robot  = len(self.ts_robot)
        self.n_wrench = len(self.ts_wrench)

        # zeroset wrenches (prefer _zeroset, fallback to raw)
        self.wrench = {}
        for fn in FINGERS:
            zkey = f'wrench_zeroset_{fn}_L'
            rkey = f'wrench_{fn}_L'
            if zkey in obs:
                self.wrench[fn] = obs[zkey][:]
            elif rkey in obs:
                self.wrench[fn] = obs[rkey][:]

    def wrench_at_step(self, robot_step):
        ts  = self.ts_robot[min(robot_step, self.n_robot-1)]
        idx = int(np.argmin(np.abs(self.ts_wrench - ts)))
        return {fn: self.wrench[fn][idx] for fn in FINGERS if fn in self.wrench}

# ── Fixed-size matplotlib canvases ────────────────────────────────────────────

class FixedCanvas(FigureCanvas):
    """Canvas that does NOT call tight_layout on every draw, keeping its size stable."""
    def __init__(self, fig):
        super().__init__(fig)
        # Prevent the canvas from shrinking/growing with content
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(10, 10)


class HandCanvas(FixedCanvas):
    def __init__(self):
        fig = Figure(facecolor='#1a1a2e')
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._style()

    def _style(self):
        ax = self.ax
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaa', labelsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor('#333355')
        for lbl in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
            lbl.set_color('#aaa')
        ax.set_xlabel('X', fontsize=7); ax.set_ylabel('Y', fontsize=7); ax.set_zlabel('Z', fontsize=7)

    def update_hand(self, urdf_hand, hand_q, wrenches):
        ax = self.ax
        ax.cla(); self._style()
        ax.scatter([0],[0],[0], color='white', s=50)

        fk = urdf_hand.all_fk(hand_q)
        for fn in FINGERS:
            c = FCOLORS[fn]
            transforms = fk[fn]
            pts = [np.zeros(3)] + [T[:3,3] for T in transforms]
            xs,ys,zs = zip(*pts)
            ax.plot(xs,ys,zs,'-o', color=c, lw=2, ms=3, label=fn)

            if wrenches and fn in wrenches:
                force = wrenches[fn][:3]
                fn_mag = float(np.linalg.norm(force))
                if fn_mag > 0.01:
                    tip = transforms[-1][:3,3]
                    R   = transforms[-1][:3,:3]
                    fd  = R @ (force/fn_mag) * fn_mag * 0.04
                    ax.quiver(tip[0],tip[1],tip[2], fd[0],fd[1],fd[2],
                              color='yellow', normalize=False,
                              arrow_length_ratio=0.3, lw=1.5)
                    ax.text(tip[0]+fd[0], tip[1]+fd[1], tip[2]+fd[2],
                            f'{fn_mag:.1f}N', color='yellow', fontsize=6)

        ax.set_xlim(-0.06, 0.26); ax.set_ylim(-0.12, 0.26); ax.set_zlim(-0.05, 0.32)
        ax.set_title('Left Hand + Force (zeroset)', color='white', fontsize=9, pad=2)
        ax.legend(loc='upper left', fontsize=6, facecolor='#1a1a2e',
                  edgecolor='#444466', labelcolor='white')
        self.draw_idle()


class ImageCanvas(FixedCanvas):
    def __init__(self, title=''):
        fig = Figure(facecolor='#1a1a2e')
        super().__init__(fig)
        self._title = title
        self.ax = fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self._imshow = None
        # Draw placeholder
        self.ax.set_facecolor('#0a0a18')
        self.ax.text(0.5, 0.5, title, color='#555577',
                     ha='center', va='center', fontsize=9,
                     transform=self.ax.transAxes)
        self.draw_idle()

    def show_image(self, img_bgr):
        ax = self.ax
        if img_bgr is None:
            return
        img_rgb = img_bgr[:,:,::-1]
        if self._imshow is None:
            ax.cla(); ax.axis('off')
            self._imshow = ax.imshow(img_rgb, aspect='auto')
            ax.set_title(self._title, color='white', fontsize=8, pad=2)
        else:
            self._imshow.set_data(img_rgb)
        self.draw_idle()


class WrenchCanvas(FixedCanvas):
    """Single plot showing selected fingers' force magnitude over time."""
    def __init__(self):
        self.fig = Figure(facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)
        self._style_ax()

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#aaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333355')
        ax.set_xlabel('Time [s]', color='#aaa', fontsize=8)
        ax.set_ylabel('Force magnitude [N]', color='#aaa', fontsize=8)
        ax.set_title('Finger Force (zeroset)', color='white', fontsize=9)

    def plot(self, demos_data, checked_demos, checked_fingers, current_step):
        ax = self.ax
        ax.cla(); self._style_ax()

        ls_list = ['-','--','-.',':']
        for di in sorted(checked_demos):
            if di >= len(demos_data): continue
            d = demos_data[di]
            for fn in checked_fingers:
                if fn not in d.wrench: continue
                w   = d.wrench[fn]             # (N_w, 6)
                t   = d.ts_wrench - d.ts_wrench[0]
                mag = np.linalg.norm(w[:, :3], axis=1)
                ls  = ls_list[di % len(ls_list)]
                lbl = f'd{di} {fn}'
                ax.plot(t, mag, color=FCOLORS[fn], lw=1.4, ls=ls,
                        alpha=0.9, label=lbl)

        # Vertical cursor line for currently displayed demo
        ref = self.current_demo_ref if hasattr(self, 'current_demo_ref') else 0
        if ref < len(demos_data) and len(demos_data) > 0:
            d0 = demos_data[ref]
            if current_step < d0.n_robot:
                ts = d0.ts_robot[current_step] - d0.ts_wrench[0]
                ax.axvline(ts, color='cyan', lw=1, ls='--', alpha=0.8)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, facecolor='#1a1a2e',
                      edgecolor='#444466', labelcolor='white',
                      loc='upper right', ncol=max(1, len(handles)//6))
        self.draw_idle()


# ── Main window ───────────────────────────────────────────────────────────────

class HDF5Viewer(QMainWindow):
    def __init__(self, hdf5_path):
        super().__init__()
        self.setWindowTitle("HDF5 Viewer — Left Hand Zeroset Wrench")
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#12121f;color:#e0e0e0;font-family:'Segoe UI',sans-serif}
            QLabel{color:#ccccdd;font-size:12px}
            QGroupBox{border:1px solid #333355;border-radius:6px;margin-top:8px;color:#aaaacc;font-size:11px;padding:4px}
            QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px}
            QSlider::groove:horizontal{background:#222244;height:6px;border-radius:3px}
            QSlider::handle:horizontal{background:#5566ff;width:16px;height:16px;border-radius:8px;margin:-5px 0}
            QPushButton{background:#2a2a4a;border:1px solid #444466;border-radius:5px;padding:5px 12px;color:#ccddff}
            QPushButton:hover{background:#3a3a6a}
            QCheckBox{color:#ccccdd;font-size:11px}
            QComboBox{background:#1e1e3a;border:1px solid #444466;border-radius:4px;color:#e0e0ff;padding:3px}
            QScrollArea{border:none}
        """)

        self.hdf5_path    = hdf5_path
        self.urdf_hand    = URDFHand(URDF_PATH)
        self.demos_data   = []
        self.current_demo = 0
        self.current_step = 0
        self.checked_demos    = {0}
        self.checked_fingers  = set(FINGERS)

        self._load_data()
        self._build_ui()
        self._refresh()

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load_data(self):
        self.demos_data = []
        with h5py.File(self.hdf5_path, 'r') as f:
            for dk in sorted(f['data'].keys()):
                self.demos_data.append(DemoData(f, dk))
        print(f"Loaded {len(self.demos_data)} demos from {self.hdf5_path}")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4); root.setContentsMargins(6,6,6,6)

        # ── Top bar ──────────────────────────────────────────────────────────
        topbar = QHBoxLayout()
        btn_open = QPushButton("📂 Open HDF5")
        btn_open.clicked.connect(self._open_file)
        topbar.addWidget(btn_open)
        topbar.addWidget(QLabel("View demo:"))
        self.combo_demo = QComboBox()
        for i,d in enumerate(self.demos_data):
            self.combo_demo.addItem(f"demo_{i}  ({d.n_robot} steps)")
        self.combo_demo.currentIndexChanged.connect(self._on_demo_changed)
        topbar.addWidget(self.combo_demo)
        topbar.addStretch()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        topbar.addWidget(self.btn_play)
        topbar.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for s in ['0.25x','0.5x','1x','2x','4x']:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentIndex(2)
        topbar.addWidget(self.speed_combo)
        root.addLayout(topbar)

        # ── Slider ───────────────────────────────────────────────────────────
        srow = QHBoxLayout()
        self.step_label = QLabel("Step: 0 / 0")
        self.step_label.setFixedWidth(110)
        srow.addWidget(self.step_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        n0 = self.demos_data[0].n_robot if self.demos_data else 1
        self.slider.setMaximum(n0 - 1)
        self.slider.valueChanged.connect(self._on_slider)
        srow.addWidget(self.slider)
        root.addLayout(srow)

        # ── Main area ────────────────────────────────────────────────────────
        main_split = QSplitter(Qt.Horizontal)

        # ── LEFT panel: demo + finger checkboxes ─────────────────────────────
        left_panel = QWidget(); left_panel.setFixedWidth(160)
        lp_layout  = QVBoxLayout(left_panel); lp_layout.setSpacing(4)

        demo_grp = QGroupBox("Demos (plot)")
        dg_lay   = QVBoxLayout(demo_grp)
        self.demo_checkboxes = []
        for i,d in enumerate(self.demos_data):
            chk = QCheckBox(f"demo_{i}")
            chk.setChecked(i == 0)
            chk.stateChanged.connect(self._on_demo_checkbox)
            dg_lay.addWidget(chk)
            self.demo_checkboxes.append(chk)
        lp_layout.addWidget(demo_grp)

        finger_grp = QGroupBox("Fingers (plot)")
        fg_lay     = QVBoxLayout(finger_grp)
        self.finger_checkboxes = {}
        for fn in FINGERS:
            chk = QCheckBox(fn)
            chk.setChecked(True)
            chk.stateChanged.connect(self._on_finger_checkbox)
            fg_lay.addWidget(chk)
            self.finger_checkboxes[fn] = chk
        lp_layout.addWidget(finger_grp)
        lp_layout.addStretch()
        main_split.addWidget(left_panel)

        # ── CENTER col: hand 3D (top) + wrench plot (bottom) ─────────────────
        center = QWidget()
        cv_lay = QVBoxLayout(center)
        cv_lay.setSpacing(4); cv_lay.setContentsMargins(0, 0, 0, 0)

        self.hand_canvas = HandCanvas()
        self.hand_canvas.setMinimumSize(380, 320)
        cv_lay.addWidget(self.hand_canvas, 3)

        self.wrench_canvas = WrenchCanvas()
        self.wrench_canvas.setMinimumSize(380, 200)
        cv_lay.addWidget(self.wrench_canvas, 2)

        main_split.addWidget(center)

        # ── RIGHT col: image_H (top) + image_T (bottom) ───────────────────────
        right = QWidget()
        rv_lay = QVBoxLayout(right)
        rv_lay.setSpacing(4); rv_lay.setContentsMargins(0, 0, 0, 0)

        self.img_H = ImageCanvas("Camera Head")
        self.img_T = ImageCanvas("Camera Table")
        self.img_H.setMinimumSize(320, 200)
        self.img_T.setMinimumSize(320, 200)
        rv_lay.addWidget(self.img_H, 1)
        rv_lay.addWidget(self.img_T, 1)

        main_split.addWidget(right)

        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 4)
        main_split.setStretchFactor(2, 3)
        root.addWidget(main_split, 1)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._play_step)
        self.resize(1500, 900)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open HDF5", "", "HDF5 (*.hdf5 *.h5)")
        if not path: return
        self.hdf5_path = path
        self._load_data()
        self.combo_demo.blockSignals(True)
        self.combo_demo.clear()
        for i,d in enumerate(self.demos_data):
            self.combo_demo.addItem(f"demo_{i}  ({d.n_robot} steps)")
        self.combo_demo.blockSignals(False)
        self.current_demo = 0; self.current_step = 0
        if self.demos_data:
            self.slider.setMaximum(self.demos_data[0].n_robot - 1)
        self._refresh()

    def _on_demo_changed(self, idx):
        self.current_demo = idx; self.current_step = 0
        n = self.demos_data[idx].n_robot if self.demos_data else 1
        self.slider.blockSignals(True)
        self.slider.setMaximum(n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._refresh()

    def _on_slider(self, val):
        self.current_step = val
        self._refresh()

    def _on_demo_checkbox(self):
        self.checked_demos = {i for i,c in enumerate(self.demo_checkboxes) if c.isChecked()}
        self._update_wrench()

    def _on_finger_checkbox(self):
        self.checked_fingers = {fn for fn,c in self.finger_checkboxes.items() if c.isChecked()}
        self._update_wrench()

    def _on_play_toggled(self, checked):
        speeds = {'0.25x':200,'0.5x':100,'1x':50,'2x':25,'4x':12}
        ms = speeds.get(self.speed_combo.currentText(), 50)
        if checked:
            self.btn_play.setText("⏸ Pause"); self.timer.start(ms)
        else:
            self.btn_play.setText("▶ Play");  self.timer.stop()

    def _play_step(self):
        if not self.demos_data: return
        n    = self.demos_data[self.current_demo].n_robot
        next_step = (self.current_step + 1) % n
        self.slider.setValue(next_step)

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self):
        d = self.demos_data[self.current_demo] if self.demos_data else None
        n = d.n_robot if d else 0
        self.step_label.setText(f"Step: {self.current_step} / {max(0,n-1)}")
        self._update_hand()
        self._update_images()
        self._update_wrench()

    def _update_hand(self):
        if not self.demos_data: return
        d    = self.demos_data[self.current_demo]
        step = min(self.current_step, d.n_robot - 1)
        wrenches = d.wrench_at_step(step)
        self.hand_canvas.update_hand(self.urdf_hand, d.hand_L[step], wrenches)

    def _update_images(self):
        if not self.demos_data: return
        d    = self.demos_data[self.current_demo]
        step = min(self.current_step, d.n_robot - 1)
        self.img_H.show_image(d.image_H[step])
        self.img_T.show_image(d.image_T[step])

    def _update_wrench(self):
        if not self.demos_data: return
        self.wrench_canvas.current_demo_ref = self.current_demo
        self.wrench_canvas.plot(
            self.demos_data,
            self.checked_demos,
            self.checked_fingers,
            self.current_step,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    hdf5 = sys.argv[1] if len(sys.argv) > 1 else HDF5_DEFAULT
    if not os.path.exists(hdf5):
        dlg = QtWidgets.QMessageBox()
        dlg.setWindowTitle("File not found")
        dlg.setText(f"HDF5 file not found:\n{hdf5}")
        dlg.exec_(); sys.exit(1)
    win = HDF5Viewer(hdf5)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
