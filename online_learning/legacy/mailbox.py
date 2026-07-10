"""
파일시스템 기반 actor <-> learner 통신 (robotmq 대체).

같은 머신 또는 공유 파일시스템을 가정한다. 모든 쓰기는 tmp 파일에 쓴 뒤 os.replace
로 원자적 교체하여, 상대가 절반만 쓰인 파일을 읽는 것을 방지한다.

  transitions/  : actor -> learner. episode HDF5 + .ready 마커.
  weights/      : learner -> actor. weights_v<N>.pt + latest.txt (버전 번호).

cross-machine으로 확장하려면 이 클래스만 ZMQ/robotmq 버전으로 교체하면 된다.
"""
import os
import glob
import json
import shutil
import tempfile

import torch


class FileMailbox:
    def __init__(self, workdir):
        self.workdir = workdir
        self.transitions_dir = os.path.join(workdir, "transitions")
        self.weights_dir = os.path.join(workdir, "weights")
        os.makedirs(self.transitions_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

    # ── actor -> learner : 에피소드 ─────────────────────────────────────────
    def send_episode(self, src_hdf5_path):
        """actor가 만든 에피소드 HDF5를 transitions로 복사하고 .ready 마커 생성."""
        counter = len(glob.glob(os.path.join(self.transitions_dir, "ep_*.hdf5")))
        name = f"ep_{counter:05d}"
        dst = os.path.join(self.transitions_dir, name + ".hdf5")
        shutil.copyfile(src_hdf5_path, dst)
        # .ready 마커 (원자적)
        ready = os.path.join(self.transitions_dir, name + ".ready")
        with open(ready, "w") as f:
            f.write("ready")
        print(f"[mailbox] 에피소드 전송: {dst}")
        return dst

    def poll_new_episodes(self):
        """아직 처리 안 된 (.ready 있고 .done 없는) 에피소드 HDF5 경로 목록 반환."""
        new_paths = []
        for ready in sorted(glob.glob(os.path.join(self.transitions_dir, "ep_*.ready"))):
            base = ready[:-len(".ready")]
            done = base + ".done"
            if not os.path.exists(done):
                new_paths.append(base + ".hdf5")
        return new_paths

    def mark_episode_done(self, hdf5_path):
        base = hdf5_path[:-len(".hdf5")]
        # .ready -> .done 로 이름 변경 (원자적)
        ready = base + ".ready"
        done = base + ".done"
        if os.path.exists(ready):
            os.replace(ready, done)

    # ── learner -> actor : 가중치 ───────────────────────────────────────────
    def publish_weights(self, payload: dict, keep_last=2):
        """payload(dict, 예: {'state_dict':..., 'version':N})를 저장하고 latest.txt 갱신."""
        version = payload["version"]
        wpath = os.path.join(self.weights_dir, f"weights_v{version}.pt")
        # tmp -> replace (원자적)
        fd, tmp = tempfile.mkstemp(dir=self.weights_dir, suffix=".pt.tmp")
        os.close(fd)
        torch.save(payload, tmp)
        os.replace(tmp, wpath)

        latest = os.path.join(self.weights_dir, "latest.txt")
        fd, tmp = tempfile.mkstemp(dir=self.weights_dir, suffix=".txt.tmp")
        with os.fdopen(fd, "w") as f:
            f.write(str(version))
        os.replace(tmp, latest)
        print(f"[mailbox] 가중치 발행: v{version} -> {wpath}")

        # 오래된 가중치 파일 정리
        allw = sorted(glob.glob(os.path.join(self.weights_dir, "weights_v*.pt")),
                      key=lambda p: int(p.split("_v")[-1].split(".pt")[0]))
        for old in allw[:-keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass

    def get_latest_weight_version(self):
        latest = os.path.join(self.weights_dir, "latest.txt")
        if not os.path.exists(latest):
            return None
        try:
            with open(latest) as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return None

    def load_weights(self, version, map_location="cpu"):
        wpath = os.path.join(self.weights_dir, f"weights_v{version}.pt")
        if not os.path.exists(wpath):
            return None
        return torch.load(wpath, map_location=map_location)

    # ── learner 상태 (GUI 데모가 표시용으로 읽음) ───────────────────────────
    def publish_status(self, status: dict):
        path = os.path.join(self.workdir, "status.json")
        fd, tmp = tempfile.mkstemp(dir=self.workdir, suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(status, f)
        os.replace(tmp, path)

    def read_status(self):
        path = os.path.join(self.workdir, "status.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (ValueError, OSError):
            return None
