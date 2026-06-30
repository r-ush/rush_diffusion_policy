"""Visualize saved action trajectories (XYZ) from actions.txt.

This script expects the log format produced by:
  np.savetxt(action_log_f, action, fmt='%.8f')

Where each action block is preceded by a comment line starting with '#'.

Example:
  # t=... iter_idx=... action_shape=(15, 7)
  <15 lines of floats>

It will plot x,y,z using the first three columns.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _parse_shape_from_header(line: str) -> tuple[int, int] | None:
	"""Extract (T, D) from a header like: '# ... action_shape=(15, 7)'."""
	m = re.search(r"action_shape=\((\d+)\s*,\s*(\d+)\)", line)
	if not m:
		return None
	return int(m.group(1)), int(m.group(2))


def load_action_txt(path: str | Path) -> list[np.ndarray]:
	"""Return list of (T,D) arrays."""
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(path)

	blocks: list[np.ndarray] = []
	curr_shape: tuple[int, int] | None = None
	curr_rows: list[list[float]] = []

	def flush():
		nonlocal curr_rows, curr_shape
		if not curr_rows:
			return
		arr = np.asarray(curr_rows, dtype=np.float32)
		if curr_shape is not None:
			# best-effort reshape check
			if arr.shape != curr_shape:
				# If shape doesn't match, keep raw; common when older logs omit shape.
				pass
		blocks.append(arr)
		curr_rows = []
		curr_shape = None

	with path.open('r') as f:
		for raw in f:
			line = raw.strip()
			if not line:
				# blank line separates blocks
				flush()
				continue
			if line.startswith('#'):
				flush()
				curr_shape = _parse_shape_from_header(line)
				continue

			# numbers line
			vals = [float(x) for x in line.split()]
			curr_rows.append(vals)

	flush()
	return blocks


def plot_blocks_xyz(
	blocks: list[np.ndarray],
	*,
	every: int = 1,
	max_blocks: int | None = None,
	show: bool = True,
	save_path: str | None = None,
):
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title('Action XYZ trajectories (first 3 dims)')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	if max_blocks is not None:
		blocks = blocks[:max_blocks]

	for i, a in enumerate(blocks):
		if a.ndim != 2 or a.shape[1] < 3:
			continue
		xyz = a[::every, :3]
		ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=1.5, label=f'blk{i}')
		ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], s=12)  # start point

	if len(blocks) <= 10:
		ax.legend(loc='best', fontsize=8)

	ax.grid(True)
	ax.set_box_aspect((1, 1, 1))

	if save_path:
		fig.savefig(save_path, dpi=200, bbox_inches='tight')
		print(f"Saved figure to {save_path}")

	if show:
		plt.show()


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--input', '-i', type=str, default='data/results/actions.txt', help='Path to actions.txt')
	p.add_argument('--every', type=int, default=1, help='Downsample: keep every Nth point')
	p.add_argument('--max_blocks', type=int, default=20, help='Plot only first N blocks')
	p.add_argument('--save', type=str, default=None, help='If set, save figure to this path (png)')
	p.add_argument('--no_show', action='store_true', help='Don\'t open an interactive window')
	args = p.parse_args()

	blocks = load_action_txt(args.input)
	print(f"Loaded {len(blocks)} action blocks from {args.input}")
	plot_blocks_xyz(
		blocks,
		every=max(1, int(args.every)),
		max_blocks=None if args.max_blocks <= 0 else int(args.max_blocks),
		show=not args.no_show,
		save_path=args.save,
	)


if __name__ == '__main__':
	main()
