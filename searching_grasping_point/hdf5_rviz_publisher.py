#!/usr/bin/env python3
"""
hdf5_rviz_publisher.py
──────────────────────
HDF5 데모 데이터를 읽어서 RViz로 손 관절 + 힘(force) 을 시각화하는 ROS2 노드.

실행 방법:
  1) RViz용 robot_state_publisher 먼저 실행:
       ros2 run robot_state_publisher robot_state_publisher \
         --ros-args -p robot_description:="$(cat /path/to/hand.urdf)"

  2) 이 스크립트 실행:
       python3 hdf5_rviz_publisher.py [HDF5_PATH] [--demo N] [--speed S] [--force-scale F]

토픽:
  /joint_states            → robot_state_publisher가 TF 생성
  /left_hand/force_markers → RViz MarkerArray (화살표 + 텍스트)

파라미터:
  HDF5_PATH    : HDF5 파일 경로 (기본값: common_data.hdf5)
  --demo N     : 재생할 demo 인덱스 (기본값: 0)
  --speed S    : 재생 속도 배율 (기본값: 1.0)
  --force-scale F : 힘 화살표 스케일 (m/N, 기본값: 0.04)
  --loop       : 재생이 끝나면 반복
  --list       : HDF5 파일의 데모 목록 출력 후 종료
"""

import sys
import os
import argparse
import time
import math
import threading

import numpy as np
import h5py

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton, QLabel, QDoubleSpinBox
from PyQt5.QtCore import Qt, QTimer

# ── ROS2 imports ──────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from builtin_interfaces.msg import Duration
    import tf2_ros
except ImportError:
    print("[ERROR] rclpy를 찾을 수 없습니다. ROS2 환경을 source하세요.")
    print("  source /opt/ros/humble/setup.bash")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
HDF5_DEFAULT = os.path.join(_HERE, "common_data.hdf5")

FINGERS  = ['thumb', 'index', 'middle', 'ring', 'baby']

# 왼손 joint 이름 (URDF 기준): left_{finger}_joint{1-4}
# 각 손가락 3개 관절 → joint4 는 joint3 mimic
LEFT_JOINT_ORDER = []
for fn in FINGERS:
    for i in range(1, 5):   # joint1 ~ joint4
        LEFT_JOINT_ORDER.append(f'left_{fn}_joint{i}')

# URDF의 손가락 끝 link (frame_id for markers)
FINGER_TIP_FRAMES = {
    'thumb':  'left_sensor_thumb',
    'index':  'left_sensor_index',
    'middle': 'left_sensor_middle',
    'ring':   'left_sensor_ring',
    'baby':   'left_sensor_baby',
}

# 손가락별 색상 (RGBA)
FINGER_COLORS = {
    'thumb':  ColorRGBA(r=0.91, g=0.30, b=0.24, a=0.9),
    'index':  ColorRGBA(r=0.20, g=0.60, b=0.86, a=0.9),
    'middle': ColorRGBA(r=0.18, g=0.80, b=0.44, a=0.9),
    'ring':   ColorRGBA(r=0.95, g=0.61, b=0.07, a=0.9),
    'baby':   ColorRGBA(r=0.61, g=0.35, b=0.71, a=0.9),
}


# ── Data loader ───────────────────────────────────────────────────────────────

class DemoData:
    """HDF5 파일에서 하나의 demo를 로드."""
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
        
        self.ts_robot  = obs['timestamp_robot'][:]  # (T,)
        self.ts_wrench = obs['timestamp_wrench'][:] # (W,)
        self.n_robot   = len(self.ts_robot)
        self.wrench    = {}
        for fn in FINGERS:
            zk = f'wrench_zeroset_{fn}_L'
            rk = f'wrench_{fn}_L'
            if zk in obs:
                self.wrench[fn] = obs[zk][:].copy()
            elif rk in obs:
                self.wrench[fn] = obs[rk][:].copy()
            
            # 확정 보정 (common_data_test.hdf5 실증 기준)
            # 엄지: X는 맞음, Y/Z가 반전 이유:
            #   thumb_joint2의 pitch 회전으로 link4 프레임이
            #   index/middle 대비 X축 기준 180도 회전되어 Y,Z 반전
            # 검지/중지: X만 반전 (raw X 부호가 반대)
            if fn in self.wrench:
                if fn == 'thumb':
                    # diag(1, -1, -1): Y, Z 반전
                    self.wrench[fn][:, 1] = -self.wrench[fn][:, 1]
                    self.wrench[fn][:, 2] = -self.wrench[fn][:, 2]
                elif fn in ('index', 'middle'):
                    # diag(1, -1, -1): Y, Z 반전
                    self.wrench[fn][:, 1] = -self.wrench[fn][:, 1]
                    self.wrench[fn][:, 2] = -self.wrench[fn][:, 2]
                # ring, baby: 추후 확인


        # Load Images
        self.image_H = obs['image_H'][:] if 'image_H' in obs else None
        self.image_T = obs['image_T'][:] if 'image_T' in obs else None

    def wrench_at_step(self, step):
        """로봇 스텝에 해당하는 wrench 값 반환."""
        ts  = self.ts_robot[min(step, self.n_robot - 1)]
        idx = int(np.argmin(np.abs(self.ts_wrench - ts)))
        return {fn: self.wrench[fn][idx] for fn in FINGERS if fn in self.wrench}

    def hand_q_at_step(self, step):
        """15차원 왼손 관절각 반환."""
        return self.hand_L[min(step, self.n_robot - 1)]


def hand_q15_to_joint20(q15):
    """
    15차원 (5손가락 × 3관절) → 20차원 (5손가락 × 4관절, joint4=joint3 mimic).
    LEFT_JOINT_ORDER와 동일한 순서.
    """
    q20 = []
    for i in range(5):
        j1, j2, j3 = float(q15[i*3]), float(q15[i*3+1]), float(q15[i*3+2])
        q20.extend([j1, j2, j3, j3])   # joint4 mirrors joint3
    return q20


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class HDF5RVizPublisher(Node):
    """
    HDF5 데모 데이터를 재생하여:
      - /joint_states        → 손 관절 상태 (RobotStatePublisher가 TF 갱신)
      - /left_hand/force_markers → 손가락 끝 힘 화살표 MarkerArray
    """

    def __init__(self, demo: DemoData, args):
        super().__init__('hdf5_rviz_publisher',
                         cli_args=['--ros-args',
                                   '-r', '/tf:=/hdf5_tf',
                                   '-r', '/tf_static:=/hdf5_tf_static'])

        self.demo          = demo
        self.force_scale   = args.force_scale
        self.loop          = args.loop
        self.rotate_axis   = getattr(args, 'rotate', 'x')
        self.step          = 0
        self._playing      = True
        self.show_plane         = True   # 접촉 평면 시각화 On/Off
        self.show_res_total     = True   # 원래 합력 화살표 On/Off
        self.show_res_normal    = True   # 법선 합력 화살표 On/Off
        self.show_res_plane     = True   # 평면 합력 화살표 On/Off
        # 엄지 힘 임계값 기반 관절 오프셋
        self.thumb_offset_enabled  = True
        self.thumb_threshold       = 4.0    # N (합력 크기)
        self.thumb_pip_offset_deg  = -5.0   # 도 (rad 변환 후 PIP에 더함)
        self.thumb_dip_offset_deg  = -10.0  # 도 (rad 변환 후 DIP에 더함)

        # Publishers
        self.js_pub = self.create_publisher(JointState, '/hdf5_joint_states', 10)
        self.mk_pub = self.create_publisher(MarkerArray, '/hdf5_force_markers', 10)

        # 재생 주기: 로봇 타임스탬프 기반, 최소 20ms
        dt_avg = 0.05  # default 20 Hz
        if demo.n_robot > 1:
            dts = np.diff(demo.ts_robot)
            dts = dts[dts > 0]
            if len(dts) > 0:
                dt_avg = float(np.median(dts))
        period_s = max(0.02, dt_avg / args.speed)

        self.timer = self.create_timer(period_s, self._tick)
        
        # TF Buffer & Listener (for sensor frame positions)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._base_frame  = 'left_hand_base_link'

        self.get_logger().info(
            f"HDF5 RViz Publisher ready  |  steps={demo.n_robot}  "
            f"period={period_s*1000:.1f}ms  force_scale={self.force_scale}"
        )
        self.get_logger().info("Topics: /hdf5_joint_states, /hdf5_force_markers")

    # ── Timer callback ────────────────────────────────────────────────────────

    def _tick(self):
        if not self._playing:
            return

        self.publish_current_step()

        # ── Step 증가 ─────────────────────────────────────────────────────────
        self.step += 1
        if self.step >= self.demo.n_robot:
            if self.loop:
                self.step = 0
            else:
                self._playing = False
                self.get_logger().info("Playback finished.")

    def publish_current_step(self):
        step = self.step
        
        # ── Wrench (엄지 힘 임계값에 쓰여야 해서 먼저 시각화) ─────────────────────────────
        wrenches = self.demo.wrench_at_step(step)
        
        # ── Joint State ──────────────────────────────────────────────────────
        q15_cmd = self.demo.hand_q_at_step(step)   # 원본 (CMD)
        q15_pub = q15_cmd.copy()                   # 수정본 (STATE, 오프셋 적용)
        
        thumb_f = wrenches.get('thumb', None)
        thumb_active = (
            self.thumb_offset_enabled
            and thumb_f is not None
            and np.linalg.norm(thumb_f[:3]) >= self.thumb_threshold
        )
        if thumb_active:
            q15_pub[1] = q15_cmd[1] + np.radians(self.thumb_pip_offset_deg)
            q15_pub[2] = q15_cmd[2] + np.radians(self.thumb_dip_offset_deg)
        
        q20 = hand_q15_to_joint20(q15_pub)

        js = JointState()
        js.header.stamp    = self.get_clock().now().to_msg()
        js.header.frame_id = ''
        js.name            = LEFT_JOINT_ORDER
        js.position        = q20

        self.js_pub.publish(js)

        # ── Force Markers ────────────────────────────────────────────────────
        ma = self._make_markers(wrenches)
        
        # ── Plane + Resultant + Projection Markers ────────────────────────────
        if self.show_plane:
            plane_ma = self._make_plane_markers(wrenches, 
                                                show_res_total=self.show_res_total,
                                                show_res_normal=self.show_res_normal,
                                                show_res_plane=self.show_res_plane)
            if plane_ma:
                ma.markers.extend(plane_ma.markers)
        else:
            # 평면 OFF: 평면 + 정사영 + 합력 마커 모두 제거
            for ns in ['contact_plane',
                       'proj_thumb', 'proj_index', 'proj_middle',
                       'normal_thumb', 'normal_index', 'normal_middle',
                       'resultant_total', 'resultant_in_plane', 'resultant_normal', 'resultant_label']:
                m = Marker()
                m.header.frame_id = self._base_frame
                m.ns     = ns
                m.id     = 200
                m.action = Marker.DELETE
                ma.markers.append(m)
        
        self.mk_pub.publish(ma)
        
        # ── GUI Callback ─────────────────────────────────────────────────────
        if hasattr(self, 'gui_callback') and self.gui_callback is not None:
            img_h = self.demo.image_H[step] if self.demo.image_H is not None else None
            img_t = self.demo.image_T[step] if self.demo.image_T is not None else None
            self.gui_callback(step, wrenches, img_h, img_t, q15_cmd, q15_pub, thumb_active)

    # ── Marker builder ────────────────────────────────────────────────────────

    def _make_markers(self, wrenches: dict) -> MarkerArray:
        """각 손가락 끝의 힘을 화살표 + 텍스트 마커로 생성."""
        marker_array = MarkerArray()
        mid = 0
        lifetime = Duration(sec=0, nanosec=0)  # 무한 지속 (다음 틱에서 덮어씀)

        for fn in FINGERS:
            if fn not in wrenches:
                continue
            force  = wrenches[fn][:3]          # fx, fy, fz
            fmag   = float(np.linalg.norm(force))
            color  = FINGER_COLORS[fn]
            frame  = FINGER_TIP_FRAMES[fn]
            scale  = self.force_scale

            if fmag < 0.01:
                # 힘이 거의 없으면 DELETE 마커로 이전 화살표 제거
                for ns in [f'force_{fn}', f'force_text_{fn}']:
                    m = Marker()
                    m.header.stamp.sec = 0
                    m.header.stamp.nanosec = 0
                    m.header.frame_id = frame
                    m.ns     = ns
                    m.id     = mid
                    m.action = Marker.DELETE
                    marker_array.markers.append(m)
                    mid += 1
                continue

            # 센서 프레임을 URDF에 정확하게 추가했으므로, 
            # 변환 없이 센서 프레임 기준으로 raw force를 바로 표시합니다.
            fx, fy, fz = float(force[0]), float(force[1]), float(force[2])

            fx_s = fx * scale
            fy_s = fy * scale
            fz_s = fz * scale

            # ── Arrow Marker ─────────────────────────────────────────────────
            arrow = Marker()
            arrow.header.stamp.sec = 0
            arrow.header.stamp.nanosec = 0
            arrow.header.frame_id = frame
            arrow.ns              = f'force_{fn}'
            arrow.id              = mid
            arrow.type            = Marker.ARROW
            arrow.action          = Marker.ADD
            arrow.lifetime        = lifetime

            start = Point(x=0.0, y=0.0, z=0.0)
            end   = Point(x=fx_s, y=fy_s, z=fz_s)
            arrow.points.append(start)
            arrow.points.append(end)

            d = 0.005  # 화살표 축 두께
            arrow.scale.x = d          # shaft diameter
            arrow.scale.y = d * 2.5    # head diameter
            arrow.scale.z = 0.0        # head length (0이면 자동)

            # 힘 크기에 따라 투명도 조절 (최대 10N → 완전 불투명)
            c = ColorRGBA(r=color.r, g=color.g, b=color.b,
                          a=min(1.0, 0.3 + 0.7 * fmag / 10.0))
            arrow.color = c

            marker_array.markers.append(arrow)
            mid += 1

            # ── Text Marker ───────────────────────────────────────────────────
            text = Marker()
            text.header.stamp.sec = 0
            text.header.stamp.nanosec = 0
            text.header.frame_id = frame
            text.ns              = f'force_text_{fn}'
            text.id              = mid
            text.type            = Marker.TEXT_VIEW_FACING
            text.action          = Marker.ADD
            text.lifetime        = lifetime

            text.pose.position.x = fx_s
            text.pose.position.y = fy_s
            text.pose.position.z = fz_s + 0.02
            text.text            = f'{fn}: {fmag:.1f}N'
            text.scale.z         = 0.015   # 텍스트 높이 (m) - 잘 보이게 키움
            text.color           = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0) # 노란색 텍스트

            marker_array.markers.append(text)
            mid += 1

        return marker_array

    # ── Plane + Projection markers ────────────────────────────────────────────

    def _make_plane_markers(self, wrenches: dict, 
                            show_res_total: bool = True, 
                            show_res_normal: bool = True, 
                            show_res_plane: bool = True):
        """thumb, index, middle 센서 원점으로 이뤄진 평면 + 정사영 + 3종류 합력 마커 생성."""
        PLANE_FINGERS = ['thumb', 'index', 'middle']
        now = rclpy.time.Time()
        
        # 1. Lookup sensor positions in base frame
        pts = {}
        for fn in PLANE_FINGERS:
            sensor_frame = FINGER_TIP_FRAMES[fn]
            try:
                t = self._tf_buffer.lookup_transform(
                    self._base_frame, sensor_frame, now,
                    timeout=rclpy.duration.Duration(seconds=0.01))
                p = t.transform.translation
                pts[fn] = np.array([p.x, p.y, p.z])
            except Exception:
                return None   # TF not ready yet
        
        if len(pts) < 3:
            return None

        ma = MarkerArray()
        lifetime = Duration(sec=0, nanosec=0)
        mid_base = 200
        
        A, B, C = pts['thumb'], pts['index'], pts['middle']
        
        # 2. Plane normal (normalised)
        AB = B - A;  AC = C - A
        n = np.cross(AB, AC)
        n_len = np.linalg.norm(n)
        if n_len < 1e-6:
            return None
        n_hat = n / n_len   # unit normal to the plane (points outward)
        
        # ── Plane marker: semi-transparent TRIANGLE_LIST ──────────────────────
        # Build a quad from the 3 sensor points + centroid offset
        centroid = (A + B + C) / 3.0
        # Extend each corner slightly outward for a larger visual plane
        scale = 1.4
        pA = centroid + (A - centroid) * scale
        pB = centroid + (B - centroid) * scale
        pC = centroid + (C - centroid) * scale
        
        plane_mk = Marker()
        plane_mk.header.stamp.sec = 0
        plane_mk.header.stamp.nanosec = 0
        plane_mk.header.frame_id = self._base_frame
        plane_mk.ns     = 'contact_plane'
        plane_mk.id     = mid_base
        plane_mk.type   = Marker.TRIANGLE_LIST
        plane_mk.action = Marker.ADD
        plane_mk.lifetime = lifetime
        plane_mk.scale.x = 1.0
        plane_mk.scale.y = 1.0
        plane_mk.scale.z = 1.0
        plane_mk.color  = ColorRGBA(r=0.4, g=0.8, b=1.0, a=0.25)
        plane_mk.pose.orientation.w = 1.0
        for p0, p1, p2 in [(pA, pB, pC), (pA, pC, pB)]:
            for p in (p0, p1, p2):
                plane_mk.points.append(Point(x=p[0], y=p[1], z=p[2]))
        ma.markers.append(plane_mk)
        mid_base += 1
        
        # ── Per-finger: projected + normal component ──────────────────────────
        PROJ_COLORS = {
            'thumb':  ColorRGBA(r=0.91, g=0.30, b=0.24, a=0.9),   # red
            'index':  ColorRGBA(r=0.20, g=0.60, b=0.86, a=0.9),   # blue
            'middle': ColorRGBA(r=0.18, g=0.80, b=0.44, a=0.9),   # green
        }
        
        for fn in PLANE_FINGERS:
            if fn not in wrenches or fn not in pts:
                continue
            
            force = wrenches[fn][:3].astype(float)
            fmag  = np.linalg.norm(force)
            if fmag < 0.01:
                continue
            
            # The force is in sensor frame; we need its direction in base frame.
            # Look up sensor frame → base frame rotation.
            sensor_frame = FINGER_TIP_FRAMES[fn]
            try:
                t = self._tf_buffer.lookup_transform(
                    self._base_frame, sensor_frame, now,
                    timeout=rclpy.duration.Duration(seconds=0.01))
                q = t.transform.rotation
                # Quaternion → rotation matrix
                qx, qy, qz, qw = q.x, q.y, q.z, q.w
                R = np.array([
                    [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
                ])
                force_world = R @ force   # force in base frame
            except Exception:
                continue
            
            scale = self.force_scale
            origin = pts[fn]
            col    = PROJ_COLORS[fn]
            
            # ── Projection onto the plane ────────────────────────────────────
            # proj_on_plane = force - dot(force, n_hat)*n_hat
            normal_component = np.dot(force_world, n_hat) * n_hat
            proj = force_world - normal_component
            
            force_end  = origin + force_world * scale
            proj_end   = origin + proj * scale
            normal_end = force_end   # normal arrow: from projection tip to force tip
            normal_start = proj_end
            
            def _arrow(ns, start_p, end_p, color, mid_id, diam=0.004):
                mk = Marker()
                mk.header.stamp.sec = 0
                mk.header.stamp.nanosec = 0
                mk.header.frame_id = self._base_frame
                mk.ns     = ns
                mk.id     = mid_id
                mk.lifetime = lifetime
                
                # Check arrow length to prevent RViz segfault
                dist = np.linalg.norm(np.array(end_p) - np.array(start_p))
                if dist < 1e-4:
                    mk.action = Marker.DELETE
                    return mk
                    
                mk.type   = Marker.ARROW
                mk.action = Marker.ADD
                mk.pose.orientation.w = 1.0
                mk.points.append(Point(x=float(start_p[0]), y=float(start_p[1]), z=float(start_p[2])))
                mk.points.append(Point(x=float(end_p[0]),   y=float(end_p[1]),   z=float(end_p[2])))
                mk.scale.x = diam;  mk.scale.y = diam * 2.5;  mk.scale.z = 0.0
                mk.color   = color
                return mk
            
            # Projected force (same color, slightly transparent)
            proj_col = ColorRGBA(r=col.r, g=col.g, b=col.b, a=0.6)
            ma.markers.append(_arrow(f'proj_{fn}', origin, proj_end, proj_col, mid_base))
            mid_base += 1
            
            # Normal component (white/bright)
            norm_col = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.85)
            ma.markers.append(_arrow(f'normal_{fn}', normal_start, normal_end, norm_col, mid_base))
            mid_base += 1
        
        # ── Resultant force arrow (sum of all 3 fingers, from centroid) ────────
        total_world = np.zeros(3)
        valid_count = 0
        for fn in PLANE_FINGERS:
            if fn not in wrenches or fn not in pts:
                continue
            force = wrenches[fn][:3].astype(float)
            if np.linalg.norm(force) < 0.01:
                continue
            sensor_frame = FINGER_TIP_FRAMES[fn]
            try:
                t = self._tf_buffer.lookup_transform(
                    self._base_frame, sensor_frame, now,
                    timeout=rclpy.duration.Duration(seconds=0.01))
                q = t.transform.rotation
                qx, qy, qz, qw = q.x, q.y, q.z, q.w
                R = np.array([
                    [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
                ])
                total_world += R @ force
                valid_count += 1
            except Exception:
                pass
        
        if valid_count > 0:
            scale = self.force_scale
            
            # 합력을 평면 내(평행) 성분과 법선 성분으로 분리
            res_normal_comp = np.dot(total_world, n_hat) * n_hat
            res_in_plane    = total_world - res_normal_comp
            
            res_total_end    = centroid + total_world * scale
            res_in_plane_end = centroid + res_in_plane * scale
            res_normal_end   = res_in_plane_end + res_normal_comp * scale
            
            # 합력: 원래 화살표 모양 (보라색)
            if show_res_total:
                res_total_col = ColorRGBA(r=0.6, g=0.2, b=0.8, a=1.0)
                ma.markers.append(_arrow('resultant_total', centroid, res_total_end,
                                         res_total_col, mid_base, diam=0.007))
            else:
                m = Marker(); m.header.frame_id = self._base_frame; m.ns = 'resultant_total'; m.id = mid_base; m.action = Marker.DELETE
                ma.markers.append(m)
            mid_base += 1
            
            # 평면 내 합력 화살표 (노란색)
            if show_res_plane:
                in_plane_col = ColorRGBA(r=1.0, g=0.95, b=0.0, a=1.0)
                ma.markers.append(_arrow('resultant_in_plane', centroid, res_in_plane_end,
                                         in_plane_col, mid_base, diam=0.007))
            else:
                m = Marker(); m.header.frame_id = self._base_frame; m.ns = 'resultant_in_plane'; m.id = mid_base; m.action = Marker.DELETE
                ma.markers.append(m)
            mid_base += 1
            
            # 법선 성분 화살표 (흰색)
            if show_res_normal:
                res_norm_col = ColorRGBA(r=0.9, g=0.9, b=0.9, a=0.9)
                ma.markers.append(_arrow('resultant_normal', res_in_plane_end, res_normal_end,
                                         res_norm_col, mid_base, diam=0.007))
            else:
                m = Marker(); m.header.frame_id = self._base_frame; m.ns = 'resultant_normal'; m.id = mid_base; m.action = Marker.DELETE
                ma.markers.append(m)
            mid_base += 1
            
            # 합력 크기 텍스트 레이블 (원래 화살표나 성분 화살표 중 하나라도 켜져 있으면 표시)
            if show_res_total or show_res_plane or show_res_normal:
                total_mag = np.linalg.norm(total_world)
                txt = Marker()
                txt.header.stamp.sec = 0
                txt.header.stamp.nanosec = 0
                txt.header.frame_id = self._base_frame
                txt.ns    = 'resultant_label'
                txt.id    = mid_base
                txt.type  = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.lifetime = lifetime
                ep = centroid + total_world * scale
                txt.pose.position.x = float(ep[0])
                txt.pose.position.y = float(ep[1])
                txt.pose.position.z = float(ep[2]) + 0.025
                txt.text  = f'\u2211F: {total_mag:.1f}N'
                txt.scale.z = 0.018
                txt.color = ColorRGBA(r=1.0, g=0.95, b=0.0, a=1.0)
                ma.markers.append(txt)
                mid_base += 1
        elif not show_res_total and not show_res_plane and not show_res_normal:
            # 합력 OFF: 이전 합력 마커 제거
            for ns in ['resultant_total', 'resultant_in_plane', 'resultant_normal', 'resultant_label']:
                m = Marker()
                m.header.frame_id = self._base_frame
                m.ns     = ns
                m.id     = mid_base
                m.action = Marker.DELETE
                ma.markers.append(m)
                mid_base += 1
        
        return ma


# ── Entry ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='HDF5 demo data → RViz (joint_states + force markers)')
    p.add_argument('hdf5',        nargs='?', default=HDF5_DEFAULT,
                   help='HDF5 파일 경로')
    p.add_argument('--demo',      type=int,   default=0,
                   help='재생할 demo 인덱스 (기본: 0)')
    p.add_argument('--speed',     type=float, default=1.0,
                   help='재생 속도 배율 (기본: 1.0)')
    p.add_argument('--force-scale', type=float, default=0.04,
                   help='힘 화살표 스케일 m/N (기본: 0.04)')
    p.add_argument('--loop',      action='store_true',
                   help='재생이 끝나면 반복')
    p.add_argument('--rotate',    type=str, choices=['none', 'x', 'y'], default='x',
                   help='Force 벡터를 90도 회전할 축 (기본: x)')
    p.add_argument('--list',      action='store_true',
                   help='데모 목록 출력 후 종료')
    return p.parse_args()


from PyQt5.QtGui import QImage, QPixmap

def cv_to_pixmap(cv_img):
    h, w, c = cv_img.shape
    bytesPerLine = 3 * w
    qImg = QImage(cv_img.data, w, h, bytesPerLine, QImage.Format_BGR888)
    # Scale down for compact UI
    return QPixmap.fromImage(qImg).scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)

class RVizPublisherGUI(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle(f"RViz Playback (Demo {node.demo.n_robot} steps)")
        self._latest_data = None
        
        main_layout = QHBoxLayout(self)
        
        # Left Panel: Controls & Force
        left_panel = QVBoxLayout()
        
        self.lbl = QLabel("Step: 0 / 0")
        left_panel.addWidget(self.lbl)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(node.demo.n_robot - 1)
        self.slider.valueChanged.connect(self.on_slider)
        left_panel.addWidget(self.slider)
        
        self.btn = QPushButton("Pause" if self.node._playing else "Play")
        self.btn.clicked.connect(self.on_play)
        left_panel.addWidget(self.btn)
        
        # --- Thumb Offset Controls ---
        offset_layout = QVBoxLayout()
        offset_layout.addWidget(QLabel("Thumb Offset Params:"))
        
        h_box1 = QHBoxLayout()
        h_box1.addWidget(QLabel("Thresh(N):"))
        self.s_thresh = QDoubleSpinBox(); self.s_thresh.setRange(0, 20); self.s_thresh.setValue(node.thumb_threshold)
        self.s_thresh.valueChanged.connect(lambda v: setattr(node, 'thumb_threshold', v))
        h_box1.addWidget(self.s_thresh)
        offset_layout.addLayout(h_box1)
        
        h_box2 = QHBoxLayout()
        h_box2.addWidget(QLabel("PIP/DIP Deg:"))
        self.s_pip = QDoubleSpinBox(); self.s_pip.setRange(-45, 45); self.s_pip.setValue(node.thumb_pip_offset_deg)
        self.s_pip.valueChanged.connect(lambda v: setattr(node, 'thumb_pip_offset_deg', v))
        self.s_dip = QDoubleSpinBox(); self.s_dip.setRange(-45, 45); self.s_dip.setValue(node.thumb_dip_offset_deg)
        self.s_dip.valueChanged.connect(lambda v: setattr(node, 'thumb_dip_offset_deg', v))
        h_box2.addWidget(self.s_pip); h_box2.addWidget(self.s_dip)
        offset_layout.addLayout(h_box2)
        left_panel.addLayout(offset_layout)
        
        self.force_lbl = QLabel("Forces (N):\n...")
        self.force_lbl.setStyleSheet("font-family: monospace;")
        left_panel.addWidget(self.force_lbl)
        
        # 평면 On/Off 토글 버튼
        self.plane_btn = QPushButton("평면: ON")
        self.plane_btn.setCheckable(True)
        self.plane_btn.setChecked(True)
        self.plane_btn.clicked.connect(self.on_plane_toggle)
        self.plane_btn.setStyleSheet(
            "QPushButton { background: #1a5c8a; color: white; font-weight: bold; padding: 4px; }"
            "QPushButton:checked { background: #1a5c8a; }"
            "QPushButton:!checked { background: #555555; }"
        )
        left_panel.addWidget(self.plane_btn)
        
        # 합력 On/Off 토글 버튼들 (가로 배치)
        res_buttons_layout = QHBoxLayout()
        
        self.res_total_btn = QPushButton("원래 합력: ON")
        self.res_total_btn.setCheckable(True)
        self.res_total_btn.setChecked(True)
        self.res_total_btn.clicked.connect(lambda c: self.on_res_toggle('total', c))
        self.res_total_btn.setStyleSheet(
            "QPushButton { background: #662288; color: white; font-weight: bold; padding: 4px; }"
            "QPushButton:checked { background: #662288; }"
            "QPushButton:!checked { background: #555555; }"
        )
        res_buttons_layout.addWidget(self.res_total_btn)
        
        self.res_norm_btn = QPushButton("법선 합력: ON")
        self.res_norm_btn.setCheckable(True)
        self.res_norm_btn.setChecked(True)
        self.res_norm_btn.clicked.connect(lambda c: self.on_res_toggle('normal', c))
        self.res_norm_btn.setStyleSheet(
            "QPushButton { background: #7a7a7a; color: white; font-weight: bold; padding: 4px; }"
            "QPushButton:checked { background: #7a7a7a; }"
            "QPushButton:!checked { background: #555555; }"
        )
        res_buttons_layout.addWidget(self.res_norm_btn)
        
        self.res_plane_btn = QPushButton("평면 합력: ON")
        self.res_plane_btn.setCheckable(True)
        self.res_plane_btn.setChecked(True)
        self.res_plane_btn.clicked.connect(lambda c: self.on_res_toggle('plane', c))
        self.res_plane_btn.setStyleSheet(
            "QPushButton { background: #7a5a00; color: white; font-weight: bold; padding: 4px; }"
            "QPushButton:checked { background: #7a5a00; }"
            "QPushButton:!checked { background: #555555; }"
        )
        res_buttons_layout.addWidget(self.res_plane_btn)
        
        left_panel.addLayout(res_buttons_layout)
        
        # Joint Angles (thumb, index, middle) -> Moved under the buttons
        left_panel.addWidget(QLabel("Joint Angles (cmd / state):"))
        self.joint_lbl = QLabel()
        self.joint_lbl.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background: #1e1e1e; color: #d4d4d4; padding: 6px;"
        )
        self.joint_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.joint_lbl.setText(self._build_joint_text(None, None))
        left_panel.addWidget(self.joint_lbl)
        
        left_panel.addStretch()
        
        main_layout.addLayout(left_panel)
        
        # Right Panel: Images
        right_panel = QVBoxLayout()
        self.img_h_lbl = QLabel()
        self.img_t_lbl = QLabel()
        right_panel.addWidget(QLabel("Head Camera:"))
        right_panel.addWidget(self.img_h_lbl)
        right_panel.addWidget(QLabel("Table Camera:"))
        right_panel.addWidget(self.img_t_lbl)
        
        main_layout.addLayout(right_panel)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)
        
        # Register GUI update callback
        self.node.gui_callback = self._on_step_published
        
        self.resize(1100, 600)
        
    def on_slider(self, val):
        self.node.step = val
        self.node.publish_current_step()
        self.lbl.setText(f"Step: {val} / {self.slider.maximum()}")
        
    def on_play(self):
        self.node._playing = not self.node._playing
        self.btn.setText("Pause" if self.node._playing else "Play")
    
    def on_plane_toggle(self, checked):
        self.node.show_plane = checked
        self.plane_btn.setText("평면: ON" if checked else "평면: OFF")
        # 평면이 OFF되면 모든 합력도 자동 OFF (비활성화)
        if not checked:
            self.res_total_btn.setChecked(False)
            self.res_norm_btn.setChecked(False)
            self.res_plane_btn.setChecked(False)
            self.on_res_toggle('total', False)
            self.on_res_toggle('normal', False)
            self.on_res_toggle('plane', False)
        
        self.res_total_btn.setEnabled(checked)
        self.res_norm_btn.setEnabled(checked)
        self.res_plane_btn.setEnabled(checked)
    
    def on_res_toggle(self, kind, checked):
        if kind == 'total':
            self.node.show_res_total = checked
            self.res_total_btn.setText("원래 합력: ON" if checked else "원래 합력: OFF")
        elif kind == 'normal':
            self.node.show_res_normal = checked
            self.res_norm_btn.setText("법선 합력: ON" if checked else "법선 합력: OFF")
        elif kind == 'plane':
            self.node.show_res_plane = checked
            self.res_plane_btn.setText("평면 합력: ON" if checked else "평면 합력: OFF")
        
    def _build_joint_text(self, q15_cmd, q15_state, thumb_active=False):
        """thumb/index/middle의 MCP/PIP/DIP 테이블.
        q15_cmd: 원본 HDF5 값 (command)
        q15_state: 오프셋 적용 후 값 (state, 작동시 리변엄지)"""
        FINGER_IDX = {'thumb': 0, 'index': 1, 'middle': 2}
        JOINT_NAMES = ['MCP', 'PIP', 'DIP']
        header = f"{'':8s}{'CMD(°)':>9s} {'STATE(°)':>9s}\n"
        header += "-" * 32 + "\n"
        
        if q15_cmd is None:
            txt = header
            for fn in FINGER_IDX:
                txt += f"{fn.upper():8s}\n"
                for jn in JOINT_NAMES:
                    txt += f"  {jn:5s}:{'---':>7s} {'---':>9s}\n"
            return txt
        
        txt = header
        for fn, fi in FINGER_IDX.items():
            # 엄지가 아닌 손가락은 cmd=state
            active = (fn == 'thumb' and thumb_active)
            label = f"[{fn.upper()}]" + (" ●" if active else "")
            txt += f"{label}\n"
            for ji, jn in enumerate(JOINT_NAMES):
                cmd_deg   = np.degrees(q15_cmd[fi * 3 + ji])
                state_deg = np.degrees(q15_state[fi * 3 + ji])
                if jn == 'DIP':
                    cmd_str = f"{'  -':>7s}"
                else:
                    cmd_str = f"{cmd_deg:7.1f}"
                # 오프셋이 적용된 콼은 가성 표시
                diff = abs(state_deg - cmd_deg)
                star = " *" if (active and jn in ('PIP', 'DIP')) else ""
                txt += f"  {jn:5s}: {cmd_str}  {state_deg:7.1f}{star}\n"
        return txt

    def _on_step_published(self, step, forces, img_h, img_t,
                           q15_cmd=None, q15_state=None, thumb_active=False):
        # ROS 스레드에서 호출됨. 데이터를 저장만 하고 GUI 업데이트는 메인 스레드(update_ui)에서 처리
        self._latest_data = (step, forces, img_h, img_t, q15_cmd, q15_state, thumb_active)
        
    def update_ui(self):
        # 1. Update slider & step label
        if self.node._playing:
            self.slider.blockSignals(True)
            self.slider.setValue(self.node.step)
            self.lbl.setText(f"Step: {self.node.step} / {self.slider.maximum()}")
            self.slider.blockSignals(False)
            
        # 2. Update images, joint table, forces from latest data
        if self._latest_data is not None:
            step, forces, img_h, img_t, q15_cmd, q15_state, thumb_active = self._latest_data
            self._latest_data = None  # Consume
            
            if img_h is not None:
                self.img_h_lbl.setPixmap(cv_to_pixmap(img_h))
            if img_t is not None:
                self.img_t_lbl.setPixmap(cv_to_pixmap(img_t))
                
            if q15_cmd is None:
                q15_cmd = self.node.demo.hand_q_at_step(step)
                q15_state = q15_cmd
            
            self.joint_lbl.setText(self._build_joint_text(q15_cmd, q15_state, thumb_active))
            self.joint_lbl.setStyleSheet(
                "font-family: monospace; font-size: 11px; "
                + ("background: #1e2a1e; color: #90ee90; padding: 6px;"
                   if thumb_active else
                   "background: #1e1e1e; color: #d4d4d4; padding: 6px;")
            )
                
            force_txt = "Forces (N):\n"
            total_mag = 0.0
            for fn, vec in forces.items():
                fx, fy, fz = vec[:3]
                mag = np.linalg.norm([fx, fy, fz])
                total_mag += mag
                force_txt += f"- {fn:6s}: {mag:5.1f} N  [X:{fx:5.1f}, Y:{fy:5.1f}, Z:{fz:5.1f}]\n"
            
            force_txt = f"Total Force: {total_mag:.1f} N\n\n" + force_txt
            self.force_lbl.setText(force_txt)


def main():
    args = parse_args()

    if not os.path.exists(args.hdf5):
        print(f"[ERROR] HDF5 파일을 찾을 수 없습니다: {args.hdf5}")
        sys.exit(1)

    # ── Load HDF5 ──────────────────────────────────────────────────────────────
    demos = []
    with h5py.File(args.hdf5, 'r') as f:
        keys = sorted(f['data'].keys())
        if args.list:
            print(f"HDF5: {args.hdf5}")
            print(f"총 {len(keys)}개 demo:")
            for i, k in enumerate(keys):
                n = len(f[f'data/{k}/observations/timestamp_robot'])
                print(f"  [{i}] {k}  ({n} steps)")
            return
        for k in keys:
            demos.append(DemoData(f, k))

    if not demos:
        print("[ERROR] HDF5에 데모가 없습니다.")
        sys.exit(1)

    if args.demo >= len(demos):
        print(f"[ERROR] demo {args.demo} 없음. 총 {len(demos)}개.")
        sys.exit(1)

    demo = demos[args.demo]
    print(f"재생: demo_{args.demo}  ({demo.n_robot} steps)  "
          f"speed={args.speed}x  force_scale={args.force_scale}m/N")
    print(f"손가락 wrench 데이터: {list(demo.wrench.keys())}")

    # ── ROS2 spin in background, PyQt in main thread ─────────────────────────
    rclpy.init()
    node = HDF5RVizPublisher(demo, args)
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    app = QApplication(sys.argv)
    gui = RVizPublisherGUI(node)
    gui.show()
    
    try:
        app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
