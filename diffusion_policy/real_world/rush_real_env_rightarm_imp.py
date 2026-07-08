from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import av
import h5py
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rightarm_interpolation_controller_imp import (
    RightarmInterpolationControllerImp,
    JOINT_TIMESERIES_KEY,
    WRIST_WRENCH_TIMESERIES_KEY,
    DOOSAN_FORCE_TIMESERIES_KEY,
    DEFAULT_WRIST_WRENCH_TOPIC,
    DEFAULT_DOOSAN_FORCE_SERVICE)
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

# NOTE: obs key 이름은 체크포인트 shape_meta가 요구하는 라벨이라 오른팔이어도 _L 유지
DEFAULT_OBS_KEY_MAP = {
    # robot
    'robot_pose_L': 'robot_pose_L',
    'robot_quat_L': 'robot_quat_L',

    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RightarmRealEnvImp:
    def __init__(self,
            # required params
            output_dir,
            robot_ip,
            # env params
            frequency=20,
            n_obs_steps=2,
            # obs
            obs_image_resolution=(320,240),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            tcp_offset=0.13,
            init_joints=False,
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=(640,480),
            # saving params
            record_raw_video=False,
            record_wrench=False,
            wrench_topic=DEFAULT_WRIST_WRENCH_TOPIC,
            record_doosan_force=False,
            doosan_force_service=DEFAULT_DOOSAN_FORCE_SERVICE,
            timeseries_hdf5_dir=None,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=False,
            multi_cam_vis_resolution=(320,240),
            # shared memory
            shm_manager=None
            ):

        # output 동영상 저장
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        # 힘/관절 timeseries HDF5 저장 위치 (episode별 1파일)
        if timeseries_hdf5_dir is None:
            timeseries_hdf5_dir = output_dir.joinpath('timeseries_hdf5')
        else:
            timeseries_hdf5_dir = pathlib.Path(timeseries_hdf5_dir)
        timeseries_hdf5_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()


        # observation용 해상도 변환
        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:   # True
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):   # bgr -> rgb, normalize, 해상도 변환
            data['color'] = color_transform(data['color'])
            return data


        # 실시간 시각화용 해상도 변환
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )
        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data


        # raw 영상 녹화
        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        # obs 영상 녹화
        if not record_raw_video:   # not False = True
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)


        # 카메라
        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
            )


        multi_cam_vis = None
        if enable_multi_cam_vis:   # 시각화
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )


        cube_diag = np.linalg.norm([1,1,1])

        if not init_joints:   # not False = True
            j_init = None

        # 125Hz 로봇 state + 힘 로깅을 get_obs 사이(수 초)까지 놓치지 않도록 버퍼 확대
        robot_get_max_k = max_obs_buffer_size
        if record_wrench or record_doosan_force:
            robot_get_max_k = max(robot_get_max_k, 512)


        # 로봇 (오른팔 임피던스 컨트롤러)
        robot = RightarmInterpolationControllerImp(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            launch_timeout=3,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            payload_mass=None,
            payload_cog=None,
            joints_init=j_init,   # None
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=robot_get_max_k,
            record_wrench=record_wrench,
            wrench_topic=wrench_topic,
            record_doosan_force=record_doosan_force,
            doosan_force_service=doosan_force_service
            )
        self.realsense = realsense
        self.robot = robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.timeseries_hdf5_dir = timeseries_hdf5_dir
        self.replay_buffer = replay_buffer
        self.record_wrench = record_wrench
        self.record_doosan_force = record_doosan_force
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None
        # 힘/관절 timeseries 로깅 버퍼
        self.wrist_wrench_records = None
        self.action_target_records = None
        self.actual_state_records = None
        self.last_timeseries_timestamp = -np.inf

        self.start_time = None

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)   # 여기서 robot.run() 돌아감!!!!
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):   # 다른 프로세스들을 기다림
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):   # 다른 프로세스들을 기다림
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):   # with문이 시작될때 __enter__ 자동 실행됨
        self.start()       # 여기서 robot.run() 돌아감
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    # observation 얻기
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data)

        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # 힘/관절 timeseries 로깅 (녹화 중일 때만 내부에서 동작)
        self._accumulate_timeseries_data(last_robot_data)
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        # 카메라 obs 데이터 얻기
        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f'image{camera_idx}'] = value['color'][this_idxs]

        # 로봇 obs 데이터 얻기
        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v

        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # accumulate obs; Accumulator 사용
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def exec_actions(self,
            actions: np.ndarray,
            timestamps: np.ndarray,
            stages: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints; input_queue에 waypoint 쌓기
        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )

        # record actions; Accumulator 사용
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
        # timeseries HDF5용 policy target action 기록
        if self.action_target_records is not None and len(new_actions) > 0:
            self.action_target_records['timestamp'].append(new_timestamps.astype(np.float64))
            self.action_target_records['elapsed_s'].append(
                (new_timestamps - self.start_time).astype(np.float64))
            self.action_target_records['receive_timestamp'].append(
                np.full((len(new_actions),), receive_time, dtype=np.float64))
            self.action_target_records['action'].append(new_actions.astype(np.float64))
            self.action_target_records['stage'].append(new_stages.astype(np.int64))


    def _accumulate_timeseries_data(self, robot_data):
        """125Hz 로봇 state 히스토리에서 아직 저장 안 한 구간을 timeseries 버퍼에 누적.
        robot_data는 controller ring buffer(get_all_state)의 반환값."""
        if self.actual_state_records is None:
            return

        timestamps = np.asarray(robot_data['robot_receive_timestamp'], dtype=np.float64)
        if len(timestamps) == 0:
            return

        # 에피소드 시작 이후 & 마지막으로 저장한 시점 이후만 (get_obs마다 중복 방지)
        mask = timestamps >= self.start_time
        mask &= timestamps > self.last_timeseries_timestamp
        if not np.any(mask):
            return

        selected_timestamps = timestamps[mask]
        self.actual_state_records['timestamp'].append(selected_timestamps)
        self.actual_state_records['elapsed_s'].append(selected_timestamps - self.start_time)

        if 'robot_pose_L' in robot_data:
            self.actual_state_records['robot_pose_L'].append(
                np.asarray(robot_data['robot_pose_L'], dtype=np.float64)[mask])
        if 'robot_quat_L' in robot_data:
            self.actual_state_records['robot_quat_L'].append(
                np.asarray(robot_data['robot_quat_L'], dtype=np.float64)[mask])
        if JOINT_TIMESERIES_KEY in robot_data:
            self.actual_state_records[JOINT_TIMESERIES_KEY].append(
                np.asarray(robot_data[JOINT_TIMESERIES_KEY], dtype=np.float64)[mask])

        # 두 힘 소스(aft 토픽 + doosan api)를 같은 timestamp 축에 함께 누적
        if (
                self.wrist_wrench_records is not None
                and (self.record_wrench or self.record_doosan_force)):
            self.wrist_wrench_records['timestamp'].append(selected_timestamps)
            self.wrist_wrench_records['elapsed_s'].append(selected_timestamps - self.start_time)
            if self.record_wrench and WRIST_WRENCH_TIMESERIES_KEY in robot_data:
                wrench = np.asarray(robot_data[WRIST_WRENCH_TIMESERIES_KEY], dtype=np.float64)
                self.wrist_wrench_records['wrench'].append(wrench[mask])
            if self.record_doosan_force and DOOSAN_FORCE_TIMESERIES_KEY in robot_data:
                dwrench = np.asarray(robot_data[DOOSAN_FORCE_TIMESERIES_KEY], dtype=np.float64)
                self.wrist_wrench_records['wrench_doosan'].append(dwrench[mask])

        self.last_timeseries_timestamp = selected_timestamps[-1]

    @staticmethod
    def _concat_record(record, key, shape=None, dtype=np.float64):
        values = record.get(key, None)
        if values is None or len(values) == 0:
            if shape is None:
                shape = (0,)
            return np.empty(shape, dtype=dtype)
        return np.concatenate(values, axis=0).astype(dtype, copy=False)

    def _save_timeseries_hdf5(self, episode_id, filename_suffix=""):
        if self.actual_state_records is None or self.action_target_records is None:
            return

        hdf5_path = self.timeseries_hdf5_dir.joinpath(
            f'episode_{episode_id:06d}{filename_suffix}.hdf5')
        with h5py.File(hdf5_path, 'w') as f:
            f.attrs['episode_id'] = episode_id
            f.attrs['start_time'] = self.start_time
            f.attrs['schema'] = (
                'action_virtual_target: policy target action (pos3+rot6d, abs); '
                'actual: measured TCP position/quaternion + 6-DoF joint angles [rad]; '
                'wrist_ft: right wrist force/torque [fx, fy, fz, tx, ty, tz] '
                'from /aft_sensor2/wrench.')

            action_group = f.create_group('action_virtual_target')
            action_group.create_dataset(
                'timestamp',
                data=self._concat_record(self.action_target_records, 'timestamp'))
            action_group.create_dataset(
                'elapsed_s',
                data=self._concat_record(self.action_target_records, 'elapsed_s'))
            action_group.create_dataset(
                'receive_timestamp',
                data=self._concat_record(self.action_target_records, 'receive_timestamp'))
            action_group.create_dataset(
                'action',
                data=self._concat_record(self.action_target_records, 'action', shape=(0, 0)),
                compression='gzip')
            action_group.create_dataset(
                'stage',
                data=self._concat_record(self.action_target_records, 'stage', dtype=np.int64))

            actual_group = f.create_group('actual')
            actual_group.create_dataset(
                'timestamp',
                data=self._concat_record(self.actual_state_records, 'timestamp'))
            actual_group.create_dataset(
                'elapsed_s',
                data=self._concat_record(self.actual_state_records, 'elapsed_s'))
            actual_group.create_dataset(
                'robot_pose_L',
                data=self._concat_record(self.actual_state_records, 'robot_pose_L', shape=(0, 3)),
                compression='gzip')
            actual_group.create_dataset(
                'robot_quat_L',
                data=self._concat_record(self.actual_state_records, 'robot_quat_L', shape=(0, 4)),
                compression='gzip')
            actual_group.create_dataset(
                'joint_R',
                data=self._concat_record(self.actual_state_records, JOINT_TIMESERIES_KEY, shape=(0, 6)),
                compression='gzip')

            # wrist_ft: 두 힘 소스를 함께 저장
            #   wrench_wrist_R  : aft 손목 F/T (/aft_sensor2/wrench, sensor frame)
            #   wrench_doosan_R : doosan api 외력 (get_tool_force, force=base / moment=tool)
            wrist_group = f.create_group('wrist_ft')
            wrist_group.attrs['columns'] = np.array(['fx', 'fy', 'fz', 'tx', 'ty', 'tz'], dtype='S')
            wrist_group.attrs['sources'] = np.array(
                ['wrench_wrist_R=aft(/aft_sensor2/wrench)',
                 'wrench_doosan_R=doosan_api(get_tool_force)'], dtype='S')
            wrist_group.create_dataset(
                'timestamp',
                data=self._concat_record(self.wrist_wrench_records, 'timestamp')
                if self.wrist_wrench_records is not None else np.empty((0,), dtype=np.float64))
            wrist_group.create_dataset(
                'elapsed_s',
                data=self._concat_record(self.wrist_wrench_records, 'elapsed_s')
                if self.wrist_wrench_records is not None else np.empty((0,), dtype=np.float64))
            wrist_group.create_dataset(
                'wrench_wrist_R',
                data=self._concat_record(self.wrist_wrench_records, 'wrench', shape=(0, 6))
                if self.wrist_wrench_records is not None else np.empty((0, 6), dtype=np.float64),
                compression='gzip')
            wrist_group.create_dataset(
                'wrench_doosan_R',
                data=self._concat_record(self.wrist_wrench_records, 'wrench_doosan', shape=(0, 6))
                if self.wrist_wrench_records is not None else np.empty((0, 6), dtype=np.float64),
                compression='gzip')
        print(f"Timeseries HDF5 saved: {hdf5_path}")


    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        # 힘/관절 timeseries 로깅 버퍼 초기화 (aft + doosan 두 소스 공용)
        if self.record_wrench or self.record_doosan_force:
            self.wrist_wrench_records = {
                'timestamp': [],
                'elapsed_s': [],
                'wrench': [],          # aft
                'wrench_doosan': []    # doosan api
            }
        else:
            self.wrist_wrench_records = None
        self.action_target_records = {
            'timestamp': [],
            'elapsed_s': [],
            'receive_timestamp': [],
            'action': [],
            'stage': []
        }
        self.actual_state_records = {
            'timestamp': [],
            'elapsed_s': [],
            'robot_pose_L': [],
            'robot_quat_L': [],
            JOINT_TIMESERIES_KEY: []
        }
        self.last_timeseries_timestamp = start_time - 1e-9
        print(f'Episode {episode_id} started!')

    def end_episode(self):
        "Stop recording"

        assert self.is_ready

        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # 마지막 구간 로봇 state(힘/관절 포함)까지 누적
            try:
                self._accumulate_timeseries_data(self.robot.get_all_state())
            except Exception as e:
                print(f"[WARNING] Failed to collect final timeseries data: {e}")

            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            episode_id = self.replay_buffer.n_episodes
            filename_suffix = "_partial"
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                filename_suffix = ""
                print(f'Episode {episode_id} saved!')

            # 힘/관절 timeseries HDF5 저장
            try:
                self._save_timeseries_hdf5(
                    episode_id=episode_id,
                    filename_suffix=filename_suffix)
            except Exception as e:
                print(f"[WARNING] Failed to save timeseries HDF5: {e}")

            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None
            self.wrist_wrench_records = None
            self.action_target_records = None
            self.actual_state_records = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        # end_episode()가 방금 저장한 힘/관절 HDF5도 같이 제거 (버려진 에피소드)
        this_hdf5 = self.timeseries_hdf5_dir.joinpath(f'episode_{episode_id:06d}.hdf5')
        if this_hdf5.exists():
            this_hdf5.unlink()
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            # stop_recording()은 자식 프로세스에 명령만 보내고 즉시 리턴(async)한다.
            # 인코더가 mp4를 다 쓰고 파일을 놓기 전에 rmtree하면 SingleRealsense가
            # FileNotFoundError로 죽는다 → 인코더 마무리를 기다린 뒤 삭제.
            self._wait_videos_released(this_video_dir)
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

    def _wait_videos_released(self, video_dir, timeout=15.0, poll=0.3):
        "인코더가 각 mp4를 다 쓰고(close) 놓을 때까지 대기 (av.open 성공 = 마무리됨)."
        deadline = time.monotonic() + timeout
        for path in sorted(video_dir.glob("*.mp4")):
            while time.monotonic() < deadline:
                try:
                    with av.open(str(path)) as c:
                        next(c.decode(video=0))
                    break
                except Exception:
                    time.sleep(poll)
