# RTX 5070 Ti + Diffusion Policy 환경 세팅 가이드

> 대상 환경: Ubuntu 22.04 / NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120) / `rush_diffusion_policy` 프로젝트

---

## 1. Anaconda 설치

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O /tmp/anaconda.sh
bash /tmp/anaconda.sh -b -p $HOME/anaconda3
$HOME/anaconda3/bin/conda init bash
source $HOME/anaconda3/etc/profile.d/conda.sh
```

---

## 2. NVIDIA 드라이버 설치

> ⚠️ RTX 5070 Ti (Blackwell)는 **open kernel 모듈** 드라이버가 필요합니다.
> `nvidia-driver-595` (proprietary)를 설치하면 `nvidia-smi`에서 "No devices were found" 오류가 발생합니다.
> 반드시 `nvidia-driver-595-open`을 사용해야 합니다.

```bash
sudo apt-get update -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update -y
sudo apt-get install -y nvidia-driver-595-open nvidia-utils-595
```

설치 후 **재부팅**:

```bash
sudo reboot
```

재부팅 후 확인:

```bash
nvidia-smi
# GPU: NVIDIA GeForce RTX 5070 Ti, Driver 595.71.05, CUDA 13.2 가 출력되어야 함
```

---

## 3. 빌드 도구 확인

```bash
sudo apt-get install -y build-essential
```

---

## 4. conda 환경 생성

`conda_environment_train.yaml`을 사용합니다.
원본 `conda_environment_real.yaml`에서 real robot 하드웨어 패키지(free-mujoco-py, pyrealsense2, spnav, ur-rtde, pynput)를 제거한 학습 전용 버전입니다.

프로젝트 디렉토리에 파일이 없는 경우 아래 내용으로 직접 생성합니다:

```bash
cat > ~/rush_diffusion_policy/conda_environment_train.yaml << 'EOF'
name: robodiff
channels:
  - pytorch
  - pytorch3d
  - nvidia
  - conda-forge
dependencies:
  - python=3.9
  - pip=22.2.2
  - cudatoolkit=11.6
  - pytorch=1.12.1
  - torchvision=0.13.1
  - pytorch3d=0.7.0
  - numpy=1.23.3
  - numba==0.56.4
  - scipy==1.9.1
  - py-opencv=4.6.0
  - cffi=1.15.1
  - ipykernel=6.16
  - matplotlib=3.6.1
  - zarr=2.12.0
  - numcodecs=0.10.2
  - h5py=3.7.0
  - hydra-core=1.2.0
  - einops=0.4.1
  - tqdm=4.64.1
  - dill=0.3.5.1
  - scikit-video=1.1.11
  - scikit-image=0.19.3
  - gym=0.21.0
  - pymunk=6.2.1
  - wandb=0.13.3
  - threadpoolctl=3.1.0
  - shapely=1.8.4
  - cython=0.29.32
  - imageio=2.22.0
  - imageio-ffmpeg=0.4.7
  - termcolor=2.0.1
  - tensorboard=2.10.1
  - tensorboardx=2.5.1
  - psutil=5.9.2
  - click=8.0.4
  - boto3=1.24.96
  - accelerate=0.13.2
  - datasets=2.6.1
  - diffusers=0.11.1
  - av=10.0.0
  - cmake=3.24.3
  - llvm-openmp=14
  - imagecodecs==2022.8.8
  - pip:
    - ray[default,tune]==2.2.0
    - pygame==2.1.2
    - pybullet-svl==3.1.6.4
    - pytorchvideo==0.1.5
    - atomics==1.0.2
    - imagecodecs==2022.9.26
EOF
```

환경 생성:

```bash
cd ~/rush_diffusion_policy
conda env create -f conda_environment_train.yaml
```

완료까지 10~20분 소요됩니다.

---

## 5. PyTorch cu128 설치

> ⚠️ RTX 5070 Ti는 sm_120(Blackwell)을 지원하는 PyTorch가 필요합니다.
> conda yaml의 pytorch 1.12.1(cu116)은 sm_120을 지원하지 않으므로 교체해야 합니다.
> cu124, cu126 빌드도 sm_90까지만 지원하므로 반드시 **cu128**을 사용해야 합니다.

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate robodiff

# conda 설치 pytorch 제거
conda remove -n robodiff pytorch torchvision pytorch3d --force -y

# cu128 버전 설치
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 6. numpy 호환성 패키지 업그레이드

PyTorch 2.8이 numpy 2.x를 설치하는데, 기존 컴파일 패키지들이 numpy 1.x용이라 충돌합니다.
numpy를 1.26.4로 고정하고 관련 패키지를 업그레이드합니다.

```bash
conda activate robodiff

# numpy 1.26.4로 고정 (2.x는 여러 패키지와 충돌)
pip install "numpy==1.26.4"

# numpy 2.x 제거로 인해 재빌드 필요한 패키지 업그레이드
pip install --upgrade h5py numba numcodecs
pip install --upgrade tensorboard tensorboardX accelerate diffusers
pip install --upgrade wandb scipy
pip install scikit-learn
```

---

## 7. 프로젝트 설치

```bash
conda activate robodiff
cd ~/rush_diffusion_policy
pip install -e .
```

---

## 8. 설치 확인

```bash
conda activate robodiff
python -c "
import torch, numpy as np, zarr, h5py, diffusers, wandb, einops, cv2, scipy
import diffusion_policy

print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('numpy:', np.__version__)
print('diffusers:', diffusers.__version__)
print('diffusion_policy: OK')

x = torch.randn(100, 100).cuda()
print('GPU 연산 테스트: OK')
"
```

**정상 출력 예시:**
```
torch: 2.8.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Ti
numpy: 1.26.4
diffusers: 0.36.0
diffusion_policy: OK
GPU 연산 테스트: OK
```

---

## 최종 환경 요약

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| GPU Driver | 595.71.05 (open kernel) |
| CUDA (driver) | 13.2 |
| Anaconda | 2024.10-1 |
| Python | 3.9 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| numpy | 1.26.4 |
| diffusers | 0.36.0 |
| wandb | 0.26.1 |
| h5py | 3.14.0 |
| zarr | 2.12.0 |

---

## 자주 발생하는 오류

| 오류 | 원인 | 해결 |
|------|------|------|
| `nvidia-smi: No devices were found` | proprietary 드라이버 사용 | `nvidia-driver-595-open`으로 재설치 후 reboot |
| `sm_120 is not compatible` | cu126 이하 PyTorch 사용 | cu128 빌드로 교체 |
| `numpy.dtype size changed` | numpy 2.x와 구버전 패키지 충돌 | `pip install "numpy==1.26.4"` 후 패키지 업그레이드 |
| `module 'numpy' has no attribute 'bool8'` | tensorboard 구버전 | `pip install --upgrade tensorboard` |
| `MUJOCO_PATH not set` | free-mujoco-py 빌드 실패 | `conda_environment_train.yaml` 사용 (mujoco 패키지 제외됨) |
