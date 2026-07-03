#!/usr/bin/env python3
"""
HDF5 Viewer with STL mesh rendering via trimesh + matplotlib Poly3DCollection.
Layout: Left=checkboxes | Center=STL hand 3D + wrench plot | Right=img_H + img_T
"""
import sys, os
import numpy as np
import h5py
import xml.etree.ElementTree as ET
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QPushButton, QCheckBox, QGroupBox,
    QComboBox, QSplitter, QFileDialog, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH  = os.path.join(_HERE, "(reference)aidin_hand_description/urdf/hand.urdf")
MESH_DIR   = os.path.join(_HERE, "(reference)aidin_hand_description/meshes/visual")
HDF5_DEFAULT = os.path.join(_HERE, "common_data.hdf5")

FINGERS = ['thumb', 'index', 'middle', 'ring', 'baby']
FCOLORS = {'thumb':'#e74c3c','index':'#3498db','middle':'#2ecc71','ring':'#f39c12','baby':'#9b59b6'}
FCOLORS_RGB = {
    'thumb':  [0.91, 0.30, 0.24],
    'index':  [0.20, 0.60, 0.86],
    'middle': [0.18, 0.80, 0.44],
    'ring':   [0.95, 0.61, 0.07],
    'baby':   [0.61, 0.35, 0.71],
}
BASE_COLOR = [0.25, 0.25, 0.30]

# ── FK helpers ────────────────────────────────────────────────────────────────

def make_T(xyz, rpy):
    T = np.eye(4)
    T[:3,:3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T

def rot_axis(axis, angle):
    ax = np.array(axis, float); ax /= np.linalg.norm(ax) + 1e-9
    R = np.eye(4); R[:3,:3] = Rotation.from_rotvec(ax * angle).as_matrix()
    return R

class URDFHand:
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        for j in root.findall('joint'):
            name = j.get('name','')
            if 'left' not in name: continue
            o  = j.find('origin')
            ae = j.find('axis')
            p  = j.find('parent'); c = j.find('child')
            self.joints[name] = {
                'type':   j.get('type'),
                'xyz':    np.array(list(map(float,(o.get('xyz','0 0 0').split()))) if o is not None else [0,0,0]),
                'rpy':    np.array(list(map(float,(o.get('rpy','0 0 0').split()))) if o is not None else [0,0,0]),
                'axis':   np.array(list(map(float,(ae.get('xyz','0 0 1').split()))) if ae is not None else [0,0,1]),
                'parent': p.get('link') if p is not None else '',
                'child':  c.get('link') if c is not None else '',
            }

    def fk_finger(self, finger, q3):
        vals  = list(q3) + [q3[-1]]
        names = [f'left_{finger}_joint{i}' for i in range(1,5)]
        T = np.eye(4); out = []
        for i, jn in enumerate(names):
            if jn not in self.joints: break
            j = self.joints[jn]
            T = T @ make_T(j['xyz'], j['rpy'])
            if j['type'] == 'revolute':
                T = T @ rot_axis(j['axis'], vals[i])
            out.append(T.copy())
        return out

    def all_fk(self, hand_q15):
        return {fn: self.fk_finger(fn, hand_q15[i*3:(i+1)*3]) for i, fn in enumerate(FINGERS)}

# ── Mesh loader ───────────────────────────────────────────────────────────────

def load_mesh(stl_name, max_faces=800):
    """Load STL and subsample faces for fast matplotlib rendering."""
    path = os.path.join(MESH_DIR, stl_name)
    if not os.path.exists(path):
        return None, None
    m = trimesh.load(path, force='mesh')
    step = max(1, len(m.faces) // max_faces)
    faces = m.faces[::step]
    return np.array(m.vertices, dtype=float), faces

# Pre-load all meshes once
_MESHES = {}
def get_meshes():
    global _MESHES
    if _MESHES: return _MESHES
    print("Loading STL meshes...")
    _MESHES['base']  = load_mesh('body_left.STL', max_faces=600)
    for fn in FINGERS:
        sfx = '_thumb' if fn == 'thumb' else ''
        _MESHES[f'{fn}_1'] = load_mesh(f'link1{sfx}.STL',  max_faces=300)
        _MESHES[f'{fn}_2'] = load_mesh(f'link2{sfx}.STL',  max_faces=300)
        _MESHES[f'{fn}_3'] = load_mesh(f'link3{sfx}.STL',  max_faces=300)
        _MESHES[f'{fn}_4'] = load_mesh(f'link4{sfx}.STL',  max_faces=300)
    print("Meshes loaded.")
    return _MESHES

def transform_mesh(verts, T):
    """Apply 4x4 transform to Nx3 vertices."""
    v4 = np.hstack([verts, np.ones((len(verts),1))])
    return (T @ v4.T).T[:,:3]


_LS = LightSource(azdeg=225, altdeg=45)   # global light source

def shaded_colors(tris, base_rgb, alpha=1.0):
    """
    Compute per-face shaded RGBA colors.
    tris : (N,3,3) array of triangle vertices (already in world coords)
    base_rgb : [r,g,b] base color 0-1
    Returns (N,4) RGBA array.
    """
    # Face normals
    e1 = tris[:,1,:] - tris[:,0,:]
    e2 = tris[:,2,:] - tris[:,0,:]
    normals = np.cross(e1, e2)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals = normals / norms

    # Light direction (fixed world-space)
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Diffuse term (0.3 ambient + 0.7 diffuse)
    diffuse = np.clip(np.dot(normals, light_dir), 0, 1)
    intensity = 0.30 + 0.70 * diffuse   # (N,)

    r, g, b = base_rgb
    colors = np.stack([
        np.clip(r * intensity, 0, 1),
        np.clip(g * intensity, 0, 1),
        np.clip(b * intensity, 0, 1),
        np.full(len(tris), 1.0) # Always fully opaque to avoid matplotlib depth sorting issues
    ], axis=1)
    return colors

# ── Data ──────────────────────────────────────────────────────────────────────

class DemoData:
    def __init__(self, f, key):
        obs = f[f'data/{key}/observations']
        hand = obs['hand_L'][:]
        
        # [Fix] 0 벡터로 들어온 오염된 데이터(initial pose)를 이전의 유효한 값으로 덮어씌움 (forward fill)
        valid_mask = np.abs(hand).sum(axis=1) > 1e-4
        if not valid_mask.all() and valid_mask.any():
            first_valid = np.argmax(valid_mask)
            if first_valid > 0:
                hand[:first_valid] = hand[first_valid]
            for i in range(1, len(hand)):
                if not valid_mask[i]:
                    hand[i] = hand[i-1]
        self.hand_L = hand
        
        self.image_H   = obs['image_H'][:]
        self.image_T   = obs['image_T'][:]
        self.ts_robot  = obs['timestamp_robot'][:]
        self.ts_wrench = obs['timestamp_wrench'][:]
        self.n_robot   = len(self.ts_robot)
        self.wrench = {}
        for fn in FINGERS:
            zk = f'wrench_zeroset_{fn}_L'; rk = f'wrench_{fn}_L'
            if zk in obs: self.wrench[fn] = obs[zk][:].copy()
            elif rk in obs: self.wrench[fn] = obs[rk][:].copy()
            
            # 확정 보정 (common_data_test.hdf5 실증 기준)
            # 엄지: X는 맞음, Y/Z가 반전 → diag(1,-1,-1)
            # 검지/중지: X만 반전 → diag(-1,1,1)
            if fn in self.wrench:
                if fn == 'thumb':
                    self.wrench[fn][:, 1] = -self.wrench[fn][:, 1]
                    self.wrench[fn][:, 2] = -self.wrench[fn][:, 2]
                elif fn in ('index', 'middle'):
                    self.wrench[fn][:, 0] = -self.wrench[fn][:, 0]
                # ring, baby: 추후 확인


    def wrench_at_step(self, step):
        ts  = self.ts_robot[min(step, self.n_robot-1)]
        idx = int(np.argmin(np.abs(self.ts_wrench - ts)))
        return {fn: self.wrench[fn][idx] for fn in FINGERS if fn in self.wrench}

# ── STL Hand Canvas ───────────────────────────────────────────────────────────

class STLHandCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor='#f0f0f0')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.0)
        self._meshes = None
        self._view_initialized = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._init_ax()

    def get_toolbar(self, parent):
        """Return a matplotlib navigation toolbar for zoom/pan."""
        return NavToolbar(self, parent)

    def _init_ax(self):
        ax = self.ax
        ax.set_facecolor('#f0f0f0')
        ax.tick_params(colors='#555555', labelsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor('#cccccc')
        ax.set_xlabel('X', fontsize=7, color='#333333')
        ax.set_ylabel('Y', fontsize=7, color='#333333')
        ax.set_zlabel('Z', fontsize=7, color='#333333')

    def update_hand(self, urdf_hand, hand_q, wrenches):
        if self._meshes is None:
            self._meshes = get_meshes()

        ax = self.ax
        
        # Save view state before clearing
        if self._view_initialized:
            azim = ax.azim
            elev = ax.elev
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()

        ax.cla(); self._init_ax()
        fk = urdf_hand.all_fk(hand_q)

        # Draw base link with shading
        bv, bf = self._meshes.get('base', (None, None))
        if bv is not None:
            step = max(1, len(bf) // 400)
            tris = bv[bf[::step]]
            colors = shaded_colors(tris, BASE_COLOR, alpha=1.0)
            poly = Poly3DCollection(tris, facecolors=colors, edgecolor='#111122', linewidth=0.1)
            ax.add_collection3d(poly)

        # Draw each finger link with shading
        for fn in FINGERS:
            transforms = fk[fn]
            c = FCOLORS_RGB[fn]
            for li, T in enumerate(transforms):
                key = f'{fn}_{li+1}'
                mv, mf = self._meshes.get(key, (None, None))
                if mv is None: continue
                tv = transform_mesh(mv, T)
                step = max(1, len(mf) // 250)
                tris = tv[mf[::step]]
                colors = shaded_colors(tris, c, alpha=1.0)
                poly = Poly3DCollection(tris, facecolors=colors, edgecolor='#111122', linewidth=0.1)
                ax.add_collection3d(poly)

            # Force arrow at tip
            if wrenches and fn in wrenches:
                tip_T = transforms[-1] if transforms else np.eye(4)
                R_link4 = tip_T[:3, :3]
                # 센서 원점 = link4 + 0.01m (local +Y, xacro tactile joint 오프셋)
                tip   = tip_T[:3, 3] + R_link4 @ np.array([0.0, 0.01, 0.0])
                force = wrenches[fn][:3]
                fn_mag = float(np.linalg.norm(force))
                if fn_mag > 0.01:
                    # 센서 프레임 → URDF link4 프레임 변환
                    # 사용자 확인 및 데이터 분석 결과:
                    # - 센서 Z축: 패드를 누를 때 음수 발생 -> -Z가 등쪽(Dorsal) -> URDF X
                    # - 센서 X축: 팁을 누를 때 양수 발생 -> -X가 팁 방향(Tip) -> URDF Y
                    # - 센서 Y축: 측면 -> 오른쪽 법칙에 맞게 보정 -> URDF Z = Y
                    force_rot = np.array([-force[2], -force[0], force[1]])
                    fd = R_link4 @ (force_rot / fn_mag) * fn_mag * 0.04
                    ax.quiver(tip[0], tip[1], tip[2],
                              fd[0], fd[1], fd[2],
                              color='magenta', normalize=False,
                              arrow_length_ratio=0.3, lw=1.5)
                    ax.text(tip[0]+fd[0], tip[1]+fd[1], tip[2]+fd[2],
                            f'{fn_mag:.1f}N', color='magenta', fontsize=6)

        if self._view_initialized:
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        else:
            ax.set_xlim(-0.07, 0.27); ax.set_ylim(-0.12, 0.25); ax.set_zlim(-0.02, 0.32)
            self._view_initialized = True

        ax.set_title('Left Hand STL + Zeroset Force', color='#333333', fontsize=9, pad=2)
        self.draw_idle()

# ── Wrench Canvas ─────────────────────────────────────────────────────────────

class WrenchCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor='#f0f0f0')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.15)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._style()
        self.current_demo_ref = 0

    def _style(self):
        ax = self.ax
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#555555', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#cccccc')
        ax.set_xlabel('Time [s]', color='#333333', fontsize=8)
        ax.set_ylabel('Force magnitude [N]', color='#333333', fontsize=8)
        ax.set_title('Finger Force (zeroset)', color='#333333', fontsize=9)

    def plot(self, demos_data, checked_demos, checked_fingers, current_step):
        ax = self.ax; ax.cla(); self._style()
        ls_list = ['-','--','-.',':']
        for di in sorted(checked_demos):
            if di >= len(demos_data): continue
            d = demos_data[di]
            for fn in checked_fingers:
                if fn not in d.wrench: continue
                t   = d.ts_wrench - d.ts_wrench[0]
                mag = np.linalg.norm(d.wrench[fn][:,:3], axis=1)
                ax.plot(t, mag, color=FCOLORS[fn], lw=1.4,
                        ls=ls_list[di % 4], alpha=0.9,
                        label=f'd{di} {fn}')
        ref = self.current_demo_ref
        if ref < len(demos_data):
            d0 = demos_data[ref]
            if current_step < d0.n_robot:
                ts = d0.ts_robot[current_step] - d0.ts_wrench[0]
                ax.axvline(ts, color='cyan', lw=1, ls='--', alpha=0.8)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, fontsize=7, facecolor='#f0f0f0',
                      edgecolor='#cccccc', labelcolor='#333333',
                      loc='upper right', ncol=max(1, len(h)//6))
        self.draw_idle()

# ── Image Canvas ──────────────────────────────────────────────────────────────

class ImageCanvas(FigureCanvas):
    def __init__(self, title=''):
        self.fig = Figure(facecolor='#f0f0f0')
        super().__init__(self.fig)
        self._title = title
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        self.ax.set_facecolor('#e0e0e0')
        self.ax.text(0.5, 0.5, title, color='#555555',
                     ha='center', va='center', transform=self.ax.transAxes)
        self._im = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.draw_idle()

    def show_image(self, img_bgr):
        if img_bgr is None: return
        rgb = img_bgr[:,:,::-1]
        if self._im is None:
            self.ax.cla(); self.ax.axis('off')
            self._im = self.ax.imshow(rgb, aspect='auto')
            self.ax.set_title(self._title, color='#333333', fontsize=8, pad=2)
        else:
            self._im.set_data(rgb)
        self.draw_idle()

# ── Main Window ───────────────────────────────────────────────────────────────

class HDF5ViewerSTL(QMainWindow):
    def __init__(self, hdf5_path):
        super().__init__()
        self.setWindowTitle("HDF5 Viewer — STL Hand + Zeroset Wrench")
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#f0f0f0;color:#333333;font-family:'Segoe UI',sans-serif}
            QLabel{color:#222222;font-size:12px}
            QGroupBox{border:1px solid #cccccc;border-radius:6px;margin-top:8px;
                      color:#444444;font-size:11px;padding:4px}
            QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px}
            QSlider::groove:horizontal{background:#dddddd;height:6px;border-radius:3px}
            QSlider::handle:horizontal{background:#5566ff;width:16px;height:16px;
                                       border-radius:8px;margin:-5px 0}
            QPushButton{background:#e0e0e0;border:1px solid #bbbbbb;border-radius:5px;
                        padding:5px 12px;color:#333333}
            QPushButton:hover{background:#d0d0d0}
            QCheckBox{color:#333333;font-size:11px}
            QComboBox{background:#ffffff;border:1px solid #cccccc;border-radius:4px;
                      color:#333333;padding:3px}
        """)
        self.hdf5_path    = hdf5_path
        self.urdf_hand    = URDFHand(URDF_PATH)
        self.demos_data   = []
        self.current_demo = 0
        self.current_step = 0
        self.checked_demos   = {0}
        self.checked_fingers = set(FINGERS)
        self._load_data()
        self._build_ui()
        self._refresh()

    def _load_data(self):
        self.demos_data = []
        with h5py.File(self.hdf5_path, 'r') as f:
            for dk in sorted(f['data'].keys()):
                self.demos_data.append(DemoData(f, dk))
        print(f"Loaded {len(self.demos_data)} demos.")

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setSpacing(4); root.setContentsMargins(6,6,6,6)

        # ── Top bar ──────────────────────────────────────────────────────────
        top = QHBoxLayout()
        btn = QPushButton("📂 Open HDF5"); btn.clicked.connect(self._open_file)
        top.addWidget(btn)
        top.addWidget(QLabel("View demo:"))
        self.combo_demo = QComboBox()
        for i,d in enumerate(self.demos_data):
            self.combo_demo.addItem(f"demo_{i}  ({d.n_robot} steps)")
        self.combo_demo.currentIndexChanged.connect(self._on_demo_changed)
        top.addWidget(self.combo_demo); top.addStretch()
        self.btn_play = QPushButton("▶ Play"); self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        top.addWidget(self.btn_play)
        top.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        for s in ['0.25x','0.5x','1x','2x','4x']: self.speed_combo.addItem(s)
        self.speed_combo.setCurrentIndex(2)
        top.addWidget(self.speed_combo)
        root.addLayout(top)

        # ── Slider ───────────────────────────────────────────────────────────
        srow = QHBoxLayout()
        self.step_label = QLabel("Step: 0 / 0"); self.step_label.setFixedWidth(110)
        srow.addWidget(self.step_label)
        self.slider = QSlider(Qt.Horizontal); self.slider.setMinimum(0)
        n0 = self.demos_data[0].n_robot if self.demos_data else 1
        self.slider.setMaximum(n0 - 1)
        self.slider.valueChanged.connect(self._on_slider)
        srow.addWidget(self.slider); root.addLayout(srow)

        # ── Main split ───────────────────────────────────────────────────────
        split = QSplitter(Qt.Horizontal)

        # Left: checkboxes
        lp = QWidget(); lp.setFixedWidth(220)
        ll = QVBoxLayout(lp); ll.setSpacing(4)
        dg = QGroupBox("Demos (plot)"); dl = QVBoxLayout(dg)
        self.demo_checkboxes = []
        for i in range(len(self.demos_data)):
            ck = QCheckBox(f"demo_{i}"); ck.setChecked(i==0)
            ck.stateChanged.connect(self._on_demo_chk); dl.addWidget(ck)
            self.demo_checkboxes.append(ck)
        ll.addWidget(dg)
        fg = QGroupBox("Fingers (plot)"); fl = QVBoxLayout(fg)
        self.finger_checkboxes = {}
        for fn in FINGERS:
            ck = QCheckBox(fn); ck.setChecked(True)
            ck.stateChanged.connect(self._on_finger_chk); fl.addWidget(ck)
            self.finger_checkboxes[fn] = ck
        ll.addWidget(fg)
        
        # Joint States display
        jg = QGroupBox("Joint States (rad)"); jl = QVBoxLayout(jg)
        self.lbl_joints = QLabel()
        self.lbl_joints.setStyleSheet("font-family: monospace; font-size: 11px;")
        jl.addWidget(self.lbl_joints)
        ll.addWidget(jg)
        
        ll.addStretch()
        split.addWidget(lp)

        # Center: STL 3D (top) + wrench (bottom)
        cc = QWidget(); cl = QVBoxLayout(cc)
        cl.setSpacing(4); cl.setContentsMargins(0,0,0,0)
        self.hand_canvas = STLHandCanvas(); self.hand_canvas.setMinimumSize(400,340)
        
        # Add toolbar for 3D navigation (zoom/pan)
        self.hand_toolbar = self.hand_canvas.get_toolbar(cc)
        cl.addWidget(self.hand_toolbar)
        cl.addWidget(self.hand_canvas, 3)
        
        self.wrench_canvas = WrenchCanvas(); self.wrench_canvas.setMinimumSize(400,200)
        cl.addWidget(self.wrench_canvas, 2)
        split.addWidget(cc)

        # Right: images stacked
        rc = QWidget(); rl = QVBoxLayout(rc)
        rl.setSpacing(4); rl.setContentsMargins(0,0,0,0)
        self.img_H = ImageCanvas("Camera Head"); self.img_H.setMinimumSize(320,200)
        self.img_T = ImageCanvas("Camera Table"); self.img_T.setMinimumSize(320,200)
        rl.addWidget(self.img_H, 1); rl.addWidget(self.img_T, 1)
        split.addWidget(rc)

        split.setStretchFactor(0,0); split.setStretchFactor(1,4); split.setStretchFactor(2,3)
        root.addWidget(split, 1)

        self.timer = QTimer(); self.timer.timeout.connect(self._play_step)
        self.resize(1500, 900)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self,"Open HDF5","","HDF5 (*.hdf5 *.h5)")
        if not path: return
        self.hdf5_path = path; self._load_data()
        self.combo_demo.blockSignals(True); self.combo_demo.clear()
        for i,d in enumerate(self.demos_data):
            self.combo_demo.addItem(f"demo_{i}  ({d.n_robot} steps)")
        self.combo_demo.blockSignals(False)
        self.current_demo=0; self.current_step=0
        if self.demos_data: self.slider.setMaximum(self.demos_data[0].n_robot-1)
        self._refresh()

    def _on_demo_changed(self, idx):
        self.current_demo=idx; self.current_step=0
        n = self.demos_data[idx].n_robot if self.demos_data else 1
        self.slider.blockSignals(True); self.slider.setMaximum(n-1)
        self.slider.setValue(0); self.slider.blockSignals(False)
        self._refresh()

    def _on_slider(self, val):
        self.current_step = val; self._refresh()

    def _on_demo_chk(self):
        self.checked_demos = {i for i,c in enumerate(self.demo_checkboxes) if c.isChecked()}
        self._update_wrench()

    def _on_finger_chk(self):
        self.checked_fingers = {fn for fn,c in self.finger_checkboxes.items() if c.isChecked()}
        self._update_wrench()

    def _on_play_toggled(self, on):
        speeds = {'0.25x':200,'0.5x':100,'1x':50,'2x':25,'4x':12}
        ms = speeds.get(self.speed_combo.currentText(), 50)
        if on: self.btn_play.setText("⏸ Pause"); self.timer.start(ms)
        else:  self.btn_play.setText("▶ Play");  self.timer.stop()

    def _play_step(self):
        if not self.demos_data: return
        n = self.demos_data[self.current_demo].n_robot
        self.slider.setValue((self.current_step+1) % n)

    def _refresh(self):
        d = self.demos_data[self.current_demo] if self.demos_data else None
        n = d.n_robot if d else 0
        self.step_label.setText(f"Step: {self.current_step} / {max(0,n-1)}")
        self._update_hand(); self._update_images(); self._update_wrench()

    def _update_hand(self):
        if not self.demos_data: return
        d = self.demos_data[self.current_demo]
        step = min(self.current_step, d.n_robot-1)
        q = d.hand_L[step]
        self.hand_canvas.update_hand(self.urdf_hand, q, d.wrench_at_step(step))
        
        # 15-DOF 관절 텍스트 업데이트
        t_str = f"T: {q[0]:5.2f} {q[1]:5.2f} {q[2]:5.2f}"
        i_str = f"I: {q[3]:5.2f} {q[4]:5.2f} {q[5]:5.2f}"
        m_str = f"M: {q[6]:5.2f} {q[7]:5.2f} {q[8]:5.2f}"
        r_str = f"R: {q[9]:5.2f} {q[10]:5.2f} {q[11]:5.2f}"
        b_str = f"B: {q[12]:5.2f} {q[13]:5.2f} {q[14]:5.2f}"
        self.lbl_joints.setText(f"{t_str}\n{i_str}\n{m_str}\n{r_str}\n{b_str}")

    def _update_images(self):
        if not self.demos_data: return
        d = self.demos_data[self.current_demo]
        step = min(self.current_step, d.n_robot-1)
        self.img_H.show_image(d.image_H[step])
        self.img_T.show_image(d.image_T[step])

    def _update_wrench(self):
        if not self.demos_data: return
        self.wrench_canvas.current_demo_ref = self.current_demo
        self.wrench_canvas.plot(self.demos_data, self.checked_demos,
                                self.checked_fingers, self.current_step)

# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    hdf5 = sys.argv[1] if len(sys.argv) > 1 else HDF5_DEFAULT
    if not os.path.exists(hdf5):
        QtWidgets.QMessageBox.critical(None, "Not found", f"HDF5 not found:\n{hdf5}")
        sys.exit(1)
    win = HDF5ViewerSTL(hdf5)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
