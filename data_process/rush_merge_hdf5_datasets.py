"""
여러 개의 diffusion policy 학습용 HDF5 (data/demo_i/... 포맷)를
하나의 HDF5로 합친다. 기존 base policy 데이터 + correction 데이터를 합쳐서
파인튜닝용 통합 데이터셋을 만들 때 사용.

사용법:
  conda activate robodiff
  python data_process/rush_merge_hdf5_datasets.py \
      --inputs /path/to/base_data.hdf5 /path/to/correction_batch1.hdf5 \
      --output /path/to/merged_finetune.hdf5
"""

import argparse
import h5py
import tqdm


def merge(input_paths, output_path):
    with h5py.File(output_path, 'w') as out_f:
        out_data = out_f.create_group('data')
        demo_idx = 0
        for path in input_paths:
            with h5py.File(path, 'r') as in_f:
                in_data = in_f['data']
                demo_names = sorted(in_data.keys(), key=lambda s: int(s.split('_')[1]))
                for name in tqdm.tqdm(demo_names, desc=f"copying {path}"):
                    in_f.copy(f"data/{name}", out_data, name=f"demo_{demo_idx}")
                    demo_idx += 1
                print(f"{path}: {len(demo_names)}개 demo 추가")

    print(f"\n완료: 총 {demo_idx}개 demo -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                         help="합칠 입력 HDF5 경로 목록 (순서대로 demo 번호가 매겨짐)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merge(args.inputs, args.output)
