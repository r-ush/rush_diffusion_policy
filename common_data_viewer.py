"""
common_data_viewer.py  —  common_data + diffusion HDF5 동시 시각화
Usage:
    /home/rush/diffusion_policy/venv_dp/bin/python3 common_data_viewer.py
"""
import sys, h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QSlider, QFrame, QTabWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# ── FK ──────────────────────────────────────────────────────────
URDF_PATH = "/home/rush/diffusion_policy/m0609.white.urdf"
try:
    _robot = rtb.ERobot.URDF(URDF_PATH); FK_AVAILABLE = True
except Exception as e:
    print(f"[WARN] FK: {e}"); FK_AVAILABLE = False

def compute_fk(joints):
    if not FK_AVAILABLE: return None, None
    tcp = _robot.fkine(joints)
    pos_mm = tcp.t * 1000.0
    euler_deg = R.from_matrix(tcp.R).as_euler('ZYX', degrees=True)
    return pos_mm, euler_deg

def rot6d_to_euler_zyx_deg(rot6d):
    """rot6d: (N,6) → euler ZYX deg (N,3)"""
    r1 = rot6d[:, :3]; r2 = rot6d[:, 3:6]
    r3 = np.cross(r1, r2)
    rotmat = np.stack([r1, r2, r3], axis=-1)  # (N,3,3)
    return R.from_matrix(rotmat).as_euler('ZYX', degrees=True)

# ── Colors ───────────────────────────────────────────────────────
BG    = "#1e1e2e"; PANEL = "#2a2a3e"; ACCENT = "#7c3aed"
TEXT  = "#e2e8f0"; BORDER= "#3f3f5f"

C_CURRENT = ["#60a5fa","#34d399","#f472b6","#facc15","#a78bfa","#fb923c"]
C_FK_POS  = ["#f87171","#4ade80","#38bdf8"]
C_FK_ROT  = ["#fb923c","#a3e635","#818cf8"]
C_ACTION  = ["#fbbf24","#86efac","#f0abfc","#fca5a5","#a5f3fc","#c4b5fd"]

LBL_POS = ["x (mm)","y (mm)","z (mm)"]
LBL_ROT = ["rz (deg)","ry (deg)","rx (deg)"]


class CommonDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Viewer — common_data + diffusion")
        self.resize(1700, 950)
        self._style()

        # common_data
        self.hdf5_common  = None
        self.cache_common = {}
        self.fk_pos = self.fk_rot = None

        # diffusion
        self.hdf5_diff   = None
        self.cache_diff  = {}

        # display flags
        self.show_current = self.show_desired = self.show_fk = True
        self.show_action  = True
        self.show_axes_pos = [True]*3
        self.show_axes_rot = [True]*3

        self._vlines = []
        self._build_ui()

    # ── style ────────────────────────────────────────────────────
    def _style(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG};color:{TEXT};font-family:'Segoe UI',sans-serif;}}
            QPushButton{{background:{ACCENT};color:white;border:none;border-radius:6px;padding:5px 12px;font-weight:bold;}}
            QPushButton:hover{{background:#6d28d9;}}
            QPushButton:disabled{{background:#4a4a6a;color:#888;}}
            QComboBox{{background:{PANEL};color:{TEXT};border:1px solid {BORDER};border-radius:4px;padding:3px 8px;}}
            QComboBox QAbstractItemView{{background:{PANEL};color:{TEXT};selection-background-color:{ACCENT};}}
            QCheckBox{{color:{TEXT};spacing:5px;}}
            QCheckBox::indicator{{width:13px;height:13px;border-radius:3px;border:1px solid {BORDER};background:{PANEL};}}
            QCheckBox::indicator:checked{{background:{ACCENT};border:1px solid {ACCENT};}}
            QSlider::groove:horizontal{{height:5px;background:{BORDER};border-radius:3px;}}
            QSlider::handle:horizontal{{background:{ACCENT};border:none;width:14px;height:14px;margin:-5px 0;border-radius:7px;}}
            QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:3px;}}
            QGroupBox{{border:1px solid {BORDER};border-radius:7px;margin-top:8px;padding:6px;font-weight:bold;color:{TEXT};}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}
            QTabWidget::pane{{border:1px solid {BORDER};background:{PANEL};}}
            QTabBar::tab{{background:{BG};color:{TEXT};padding:6px 14px;border-radius:4px 4px 0 0;}}
            QTabBar::tab:selected{{background:{ACCENT};}}
            QLabel{{color:{TEXT};}}
        """)

    # ── UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(5); rl.setContentsMargins(8,8,8,8)

        # ── file bars ───────────────────────────────────────────
        fb = QHBoxLayout()
        self.btn_open_c = QPushButton("📂 common_data HDF5")
        self.btn_open_c.clicked.connect(lambda: self._open('common'))
        self.lbl_common = QLabel("No file"); self.lbl_common.setStyleSheet("color:#94a3b8;font-size:11px;")

        self.btn_open_d = QPushButton("📂 diffusion HDF5")
        self.btn_open_d.clicked.connect(lambda: self._open('diff'))
        self.lbl_diff = QLabel("No file"); self.lbl_diff.setStyleSheet("color:#94a3b8;font-size:11px;")

        for w in [self.btn_open_c, self.lbl_common,
                  QLabel("  |  "), self.btn_open_d, self.lbl_diff]:
            fb.addWidget(w)
        fb.addStretch()
        rl.addLayout(fb)

        # ── demo bar ────────────────────────────────────────────
        db = QHBoxLayout()
        db.addWidget(QLabel("Demo:"))
        self.combo_demo = QComboBox(); self.combo_demo.setEditable(True); self.combo_demo.setMinimumWidth(120)
        self.combo_demo.currentIndexChanged.connect(self._on_demo_changed)
        self.combo_demo.lineEdit().returnPressed.connect(self._on_demo_text)
        self.btn_prev = QPushButton("◀"); self.btn_prev.setMaximumWidth(40)
        self.btn_next = QPushButton("▶"); self.btn_next.setMaximumWidth(40)
        self.btn_prev.clicked.connect(lambda: self.combo_demo.setCurrentIndex(max(0,self.combo_demo.currentIndex()-1)))
        self.btn_next.clicked.connect(lambda: self.combo_demo.setCurrentIndex(min(self.combo_demo.count()-1,self.combo_demo.currentIndex()+1)))
        self.lbl_info = QLabel(""); self.lbl_info.setStyleSheet("color:#94a3b8;font-size:11px;")
        for w in [self.combo_demo, self.btn_prev, self.btn_next, self.lbl_info]:
            db.addWidget(w)
        db.addStretch()
        rl.addLayout(db)

        # ── splitter ────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal); rl.addWidget(splitter)

        # ── Left panel ──────────────────────────────────────────
        left = QWidget(); left.setMaximumWidth(250)
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,6,0)

        # Data source checkboxes
        src = QGroupBox("Data Sources")
        sl = QVBoxLayout()
        self.cb_current  = self._cb("Current Pose (common)", True, sl)
        self.cb_desired  = self._cb("Desired Pose (common)", True, sl)
        self.cb_fk       = self._cb("FK Pose (joint_L)", FK_AVAILABLE, sl, enabled=FK_AVAILABLE)
        self.cb_action   = self._cb("Action pos (diffusion)", True, sl)
        self.cb_obs_pose = self._cb("Obs robot_pose_L (diff)", True, sl)
        src.setLayout(sl); ll.addWidget(src)

        # Axis checkboxes
        ax_grp = QGroupBox("Axes")
        al = QVBoxLayout()
        self.cb_pos = self._cb("▶ Position", True, al, connect=self._pos_group)
        indent = "margin-left:14px;color:#94a3b8;"
        self.cb_px = self._cb("x", True, al, style=indent)
        self.cb_py = self._cb("y", True, al, style=indent)
        self.cb_pz = self._cb("z", True, al, style=indent)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet(f"color:{BORDER};")
        al.addWidget(sep)
        self.cb_rot = self._cb("▶ Rotation", True, al, connect=self._rot_group)
        self.cb_rrz = self._cb("rz", True, al, style=indent)
        self.cb_rry = self._cb("ry", True, al, style=indent)
        self.cb_rrx = self._cb("rx", True, al, style=indent)
        al.addStretch(); ax_grp.setLayout(al); ll.addWidget(ax_grp)

        self.lbl_stats = QLabel(""); self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setStyleSheet("font-size:11px;color:#94a3b8;padding:4px;")
        ll.addWidget(self.lbl_stats)

        # images
        img_grp = QGroupBox("Images")
        il = QVBoxLayout()
        self.lbl_img_c = QLabel("No image"); self.lbl_img_c.setAlignment(Qt.AlignCenter)
        self.lbl_img_c.setMinimumSize(200,150); self.lbl_img_c.setStyleSheet(f"background:{PANEL};border-radius:5px;")
        self.lbl_img_d = QLabel("No image"); self.lbl_img_d.setAlignment(Qt.AlignCenter)
        self.lbl_img_d.setMinimumSize(200,150); self.lbl_img_d.setStyleSheet(f"background:{PANEL};border-radius:5px;")
        il.addWidget(QLabel("common_data:")); il.addWidget(self.lbl_img_c)
        il.addWidget(QLabel("diffusion:")); il.addWidget(self.lbl_img_d)
        img_grp.setLayout(il); ll.addWidget(img_grp)
        splitter.addWidget(left)

        # ── Right panel ─────────────────────────────────────────
        right = QWidget(); rr = QVBoxLayout(right); rr.setContentsMargins(0,0,0,0)
        self.figure = Figure(facecolor=BG)
        self.canvas = FigureCanvas(self.figure)
        rr.addWidget(self.canvas, stretch=1)

        sr = QHBoxLayout(); sr.addWidget(QLabel("Step:"))
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_step = QLabel("0/0"); self.lbl_step.setMinimumWidth(70)
        sr.addWidget(self.slider); sr.addWidget(self.lbl_step)
        rr.addLayout(sr)
        splitter.addWidget(right)
        splitter.setSizes([240,1460])

    def _cb(self, text, checked, layout, connect=None, enabled=True, style=None):
        cb = QCheckBox(text); cb.setChecked(checked); cb.setEnabled(enabled)
        if style: cb.setStyleSheet(style)
        cb.stateChanged.connect(connect if connect else self._on_opt)
        layout.addWidget(cb); return cb

    # ── file open ────────────────────────────────────────────────
    def _open(self, kind):
        path, _ = QFileDialog.getOpenFileName(self, "Open HDF5","","HDF5 (*.hdf5 *.h5);;All (*)")
        if path: self.load_file(path, kind)

    def load_file(self, path, kind='common'):
        try:
            f = h5py.File(path, 'r')
            if kind == 'common':
                if self.hdf5_common: self.hdf5_common.close()
                self.hdf5_common = f
                self.lbl_common.setText(path.split('/')[-1])
            else:
                if self.hdf5_diff: self.hdf5_diff.close()
                self.hdf5_diff = f
                self.lbl_diff.setText(path.split('/')[-1])
            self._refresh_demo_list()
        except Exception as e:
            print(f"[ERR] {e}")

    def _refresh_demo_list(self):
        # merge demo keys from both files
        keys = set()
        for f in [self.hdf5_common, self.hdf5_diff]:
            if f and 'data' in f: keys |= set(f['data'].keys())
        keys = sorted(keys, key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
        self.combo_demo.blockSignals(True)
        self.combo_demo.clear(); self.combo_demo.addItems(keys)
        self.combo_demo.blockSignals(False)
        self.lbl_info.setText(f"({len(keys)} demos)")
        if keys:
            self.combo_demo.setCurrentIndex(0); self._on_demo_changed()

    def _on_demo_text(self):
        t = self.combo_demo.currentText()
        i = self.combo_demo.findText(t)
        if i < 0 and t.isdigit(): i = self.combo_demo.findText(f"demo_{t}")
        if i >= 0: self.combo_demo.setCurrentIndex(i)

    # ── data loading ─────────────────────────────────────────────
    def _on_demo_changed(self):
        demo = self.combo_demo.currentText()
        if not demo: return
        self.cache_common = {}; self.cache_diff = {}
        self.fk_pos = self.fk_rot = None

        # common_data
        if self.hdf5_common and 'data' in self.hdf5_common and demo in self.hdf5_common['data']:
            obs = self.hdf5_common['data'][demo]['observations']
            cp_raw = np.asarray(obs['current_pose'])   # (N_w, 6)
            dp_raw = np.asarray(obs['desired_pose'])   # (N_w, 6)
            jL     = np.asarray(obs['joint_L'])        # (N_r, 6)
            ts_r   = np.asarray(obs['timestamp_robot'])
            ts_w   = np.asarray(obs['timestamp_wrench'])
            # align to robot timestamps
            idx = np.array([np.argmin(np.abs(ts_w - t)) for t in ts_r])
            self.cache_common = {
                'current_pose': cp_raw[idx],
                'desired_pose': dp_raw[idx],
                'joint_L': jL,
                'timestamp_robot': ts_r,
                'aligned_indices': idx,
                'n': len(ts_r),
            }
            if 'image_H' in obs: self.cache_common['image_H'] = obs['image_H']
            if FK_AVAILABLE:
                self.fk_pos, self.fk_rot = compute_fk(jL)

        # diffusion
        if self.hdf5_diff and 'data' in self.hdf5_diff and demo in self.hdf5_diff['data']:
            d = self.hdf5_diff['data'][demo]
            actions = np.asarray(d['actions'])          # (N,9) trans(3)+6d_rot(6)
            obs_d   = d['obs']
            # action pos: m→mm, rot: 6d→euler ZYX deg
            act_pos_mm  = actions[:, :3] * 1000.0
            act_rot_deg = rot6d_to_euler_zyx_deg(actions[:, 3:9])
            self.cache_diff = {
                'act_pos_mm':  act_pos_mm,
                'act_rot_deg': act_rot_deg,
                'robot_pose_L': np.asarray(obs_d['robot_pose_L']) * 1000.0,  # m→mm
                'n': len(actions),
            }
            if 'image0' in obs_d: self.cache_diff['image0'] = obs_d['image0']

        # slider: use common N_r if available, else diffusion N
        n = self.cache_common.get('n') or self.cache_diff.get('n', 0)
        self.slider.setEnabled(n > 0)
        self.slider.setMinimum(0); self.slider.setMaximum(max(0,n-1)); self.slider.setValue(0)

        self._update_stats(); self._update_plot()
        self._update_image_c(0); self._update_image_d(0)

    # ── options ──────────────────────────────────────────────────
    def _pos_group(self):
        v = self.cb_pos.isChecked()
        for cb in (self.cb_px, self.cb_py, self.cb_pz):
            cb.blockSignals(True); cb.setChecked(v); cb.blockSignals(False)
        self._on_opt()

    def _rot_group(self):
        v = self.cb_rot.isChecked()
        for cb in (self.cb_rrz, self.cb_rry, self.cb_rrx):
            cb.blockSignals(True); cb.setChecked(v); cb.blockSignals(False)
        self._on_opt()

    def _on_opt(self):
        self.show_current  = self.cb_current.isChecked()
        self.show_desired  = self.cb_desired.isChecked()
        self.show_fk       = self.cb_fk.isChecked()
        self.show_action   = self.cb_action.isChecked()
        self.show_obs_pose = self.cb_obs_pose.isChecked()
        self.show_axes_pos = [self.cb_px.isChecked(), self.cb_py.isChecked(), self.cb_pz.isChecked()]
        self.show_axes_rot = [self.cb_rrz.isChecked(), self.cb_rry.isChecked(), self.cb_rrx.isChecked()]
        self._update_plot()

    # ── slider ───────────────────────────────────────────────────
    def _on_slider(self):
        s = self.slider.value(); m = self.slider.maximum()
        self.lbl_step.setText(f"{s}/{m}")
        for vl in self._vlines: vl.set_xdata([s, s])
        self.canvas.draw_idle()
        self._update_image_c(s); self._update_image_d(s)

    # ── plot helpers ─────────────────────────────────────────────
    def _make_ax(self, gs, row, title):
        ax = self.figure.add_subplot(gs[row])
        ax.set_facecolor(PANEL); ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.set_title(title, color=TEXT, fontsize=9, pad=3)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)
        return ax

    def _vline(self, ax):
        vl = ax.axvline(x=self.slider.value(), color='#f472b6', lw=1.5, alpha=0.8)
        self._vlines.append(vl)

    # ── plot ─────────────────────────────────────────────────────
    def _update_plot(self):
        self.figure.clear(); self._vlines = []
        if not self.cache_common and not self.cache_diff:
            self.canvas.draw(); return

        cc = self.cache_common; cd = self.cache_diff
        pos_on = self.show_axes_pos; rot_on = self.show_axes_rot

        pos_rows = [(i, LBL_POS[i]) for i in range(3) if pos_on[i]]
        rot_rows = [(i, LBL_ROT[i]) for i in range(3) if rot_on[i]]
        n_rows = len(pos_rows) + len(rot_rows)
        if n_rows == 0: self.canvas.draw(); return

        hsp = max(0.2, 0.55 - 0.04*n_rows)
        top = 0.96 if n_rows >= 4 else 0.93
        bot = 0.05 if n_rows >= 4 else 0.08
        gs = gridspec.GridSpec(n_rows, 1, figure=self.figure,
                               hspace=hsp, left=0.08, right=0.97, top=top, bottom=bot)
        row = 0

        # common x-axis lengths
        xc = np.arange(cc['n']) if cc else None
        xd = np.arange(cd['n']) if cd else None

        # obs robot_pose_L 색상: current 계열보다 밝은 마젠타 계열
        C_OBS_POSE = ["#e879f9", "#a3e635", "#67e8f9"]  # pos x,y,z
        for i, lbl in pos_rows:
            ax = self._make_ax(gs, row, f"Position — {lbl}")
            if cc and xc is not None:
                if self.show_current:
                    ax.plot(xc, cc['current_pose'][:,i], color=C_CURRENT[i], lw=1.5, label="current", alpha=0.9)
                if self.show_desired:
                    ax.plot(xc, cc['desired_pose'][:,i], color=C_CURRENT[i], lw=1.5, label="desired", ls='--', alpha=0.75)
                if self.show_fk and self.fk_pos is not None:
                    ax.plot(xc, self.fk_pos[:,i], color=C_FK_POS[i], lw=1.2, label="FK", ls=':', alpha=0.9)
            if cd and xd is not None:
                if self.show_action:
                    ax.plot(xd, cd['act_pos_mm'][:,i], color=C_ACTION[i], lw=1.5, label="action", ls='-.', alpha=0.9)
                if getattr(self, 'show_obs_pose', True) and 'robot_pose_L' in cd:
                    ax.plot(xd, cd['robot_pose_L'][:,i], color=C_OBS_POSE[i], lw=1.2, label="obs_pose", ls=(0,(3,1,1,1)), alpha=0.9)
            ax.legend(loc='upper right', fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            self._vline(ax); row += 1

        for i, lbl in rot_rows:
            ax = self._make_ax(gs, row, f"Rotation — {lbl}")
            if cc and xc is not None:
                if self.show_current:
                    ax.plot(xc, cc['current_pose'][:,3+i], color=C_CURRENT[3+i], lw=1.5, label="current", alpha=0.9)
                if self.show_desired:
                    ax.plot(xc, cc['desired_pose'][:,3+i], color=C_CURRENT[3+i], lw=1.5, label="desired", ls='--', alpha=0.75)
                if self.show_fk and self.fk_rot is not None:
                    ax.plot(xc, self.fk_rot[:,i], color=C_FK_ROT[i], lw=1.2, label="FK", ls=':', alpha=0.9)
            if cd and xd is not None and self.show_action:
                ax.plot(xd, cd['act_rot_deg'][:,i], color=C_ACTION[3+i], lw=1.5, label="action", ls='-.', alpha=0.9)
            ax.legend(loc='upper right', fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
            self._vline(ax); row += 1

        self.canvas.draw()

    # ── images ───────────────────────────────────────────────────
    def _show_img(self, lbl, arr, idx):
        if arr is None: lbl.setText("No image"); return
        idx = min(idx, arr.shape[0]-1)
        frame = np.ascontiguousarray(np.asarray(arr[idx]).astype(np.uint8))
        # BGR→RGB if needed (common_data is BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = frame[...,::-1].copy()
        h,w,c = frame.shape
        qi = QImage(frame.data, w, h, c*w, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qi).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_image_c(self, idx):
        self._show_img(self.lbl_img_c, self.cache_common.get('image_H'), idx)

    def _update_image_d(self, idx):
        self._show_img(self.lbl_img_d, self.cache_diff.get('image0'), idx)

    # ── stats ────────────────────────────────────────────────────
    def _update_stats(self):
        lines = []
        if self.cache_common:
            n = self.cache_common['n']
            ai = self.cache_common.get('aligned_indices')
            trim = f" (wrench {ai[0]}~{ai[-1]})" if ai is not None else ""
            lines.append(f"<b>common</b> N_r={n}{trim}")
        if self.cache_diff:
            lines.append(f"<b>diffusion</b> N={self.cache_diff['n']}")
        self.lbl_stats.setText("<br>".join(lines))

    def closeEvent(self, event):
        for f in [self.hdf5_common, self.hdf5_diff]:
            if f: f.close()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = CommonDataViewer()

    COMMON_PATH = '/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260518_0136/common_data.hdf5'
    DIFF_PATH   = '/media/rush/00d3eaaf-732e-4a7f-8bd3-6fee68d14fe7/260518_0136/diffusion_plug_desired_pose_only20.hdf5'
    viewer.load_file(COMMON_PATH, 'common')
    viewer.load_file(DIFF_PATH,   'diff')

    viewer.show()
    sys.exit(app.exec_())
