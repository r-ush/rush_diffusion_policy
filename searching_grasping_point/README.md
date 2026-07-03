# HDF5 Tactile Force & Kinematics Visualization Toolkit

이 레포지토리는 로봇 손(다지 다관절 핸드)의 렌치(Wrench/Tactile) 센서 데이터와 관절 각도(Kinematics)를 HDF5 파일로부터 읽어와 RViz 및 STL 뷰어에서 3D 화살표로 렌더링하기 위한 툴킷입니다. 

## 주요 기능 및 개발 내용

### 1. 격리된 ROS2 RViz 플레이백 시스템 (`run_hdf5_rviz.sh`)
- 라이브 로봇의 토픽과 충돌을 피하기 위해 `/hdf5_` prefix를 적용한 전용 TF 트리를 구축했습니다.
- `robot_state_publisher`와 `rviz2`에 `/tf` 및 `/tf_static` 맵핑을 적용하여 HDF5 시뮬레이션 데이터를 안전하게 시각화합니다.
- `Ctrl+C` 입력 시 모든 백그라운드 프로세스(RVIZ, RSP)가 안전하게 종료되도록 클린업 스크립트를 적용했습니다.

### 2. RViz HDF5 데이터 퍼블리셔 (`hdf5_rviz_publisher.py`)
- HDF5 파일의 로봇 스텝과 렌치 센서의 타임스탬프를 동기화하여 재생합니다.
- **결측 데이터(Data Corruption) 필터링:** HDF5 데이터 기록 중 간헐적으로 발생하는 빈(zero-vector) joint 데이터로 인해 발생하는 손떨림(jitter)을 방지하기 위해 Forward-fill(최근 유효값 복사) 로직을 추가했습니다.
- **안전한 화살표 렌더링:** 힘 벡터의 크기가 0.1mm보다 작을 때(거의 0일 때) 발생하는 RViz의 Segmentation Fault 크래시를 방지하기 위해 크기가 작을 시 `Marker.DELETE` 액션을 발행하여 예외 처리했습니다.
- RViz Text Marker를 추가해 화살표와 함께 직관적인 N(뉴턴) 단위 힘 크기를 시각적으로 확인 가능합니다.

### 3. URDF 센서 프레임 표준화 (`hand_abs_with_sensors.urdf`)
- `left_link4_{fn}`에서 `left_sensor_{fn}`으로 향하는 Sensor Frame(기준 좌표계)을 추가했습니다.
- 모든 센서 프레임의 `rpy`를 일관성 있게 `1.5707963267948966 0.0 -1.5707963267948966`로 고정하여 복잡했던 렌더링 축 연산을 하나로 통일했습니다.

### 4. 실증 기반 Wrench Vector 캘리브레이션
- 실제 테스트 환경에서 각각의 손가락(엄지, 검지, 중지)에 +X, +Y, +Z 방향으로 직접 물리적 힘을 가하는 데모 데이터를 수집/분석했습니다.
- HDF5의 Raw Data와 URDF 기구학(Kinematics) 간의 오차를 분석하여 **최종적으로 X축은 유지하고, Y축과 Z축을 반전시키는 `diag(1, -1, -1)` 보정 매트릭스**를 엄지, 검지, 중지에 일관성 있게 적용하여 RViz의 물리적 방향과 완벽하게 일치시켰습니다.

### 5. 독립적인 3D STL 뷰어 (`hdf5_viewer_stl.py`)
- RViz 없이도 HDF5 파일을 빠르게 스크러빙(Scrubbing)하며 시각화할 수 있는 Open3D / Trimesh 기반 뷰어를 개선했습니다.
- 타임라인 재생 중 줌/팬 등의 카메라 시점이 초기화되지 않도록 상태(State) 영속성을 적용했습니다.
- Wireframe 렌더링을 Phong 쉐이딩 솔리드 렌더링과 시인성이 높은 라이트 그레이(Light-gray) UI 톤으로 개선했습니다.

## 실행 방법

### RViz 시각화
```bash
./run_hdf5_rviz.sh
```

### STL 뷰어 시각화
```bash
python3 hdf5_viewer_stl.py [HDF5_FILE] --demo 0
```
