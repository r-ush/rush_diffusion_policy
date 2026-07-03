#!/usr/bin/env python3
"""
Compare two action-denoising logs that share the same initial noise trajectory.

Typical pair:
    action_denoising_traj_{i}.txt
    action_denoising_traj_{i}_change_obs_30.txt

The visualization focuses on xyz only and makes it easy to see:
1. when the two denoising runs start to diverge,
2. how large the branch gap becomes,
3. how each branch continues denoising after the condition change.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap


HEADER_RE = re.compile(
    r"^(initial(?:\s+noise)?\s+trajectory:|timestep\s+(-?\d+),\s*trajectory:)\s*$",
    flags=re.MULTILINE,
)

ORIGINAL_BASE_COLOR = "#1E5BD9"
CHANGED_BASE_COLOR = "#E67E22"
GAP_BASE_COLOR = "#374151"
BRANCH_HIGHLIGHT_COLOR = "#D97706"


@dataclass(frozen=True)
class ComparisonData:
    original_xyz: np.ndarray
    changed_xyz: np.ndarray
    labels: List[str]
    original_step_disp: np.ndarray
    changed_step_disp: np.ndarray
    branch_gap: np.ndarray
    branch_start_idx: int | None


@dataclass
class ComparisonRenderState:
    fig: plt.Figure
    ax3d_original: object
    ax3d_changed: object
    ax_gap: object
    ax_move: object
    original_lines: List[object]
    changed_lines: List[object]
    original_scatter: object
    changed_scatter: object
    original_info_text: object
    changed_info_text: object
    gap_cursor: object
    gap_cursor_dot: object
    move_cursor: object
    original_move_dot: object
    changed_move_dot: object


def _extract_balanced(text: str, open_idx: int, open_char: str, close_char: str) -> int:
    """Return the matching closing-char index for a balanced expression."""
    depth = 0
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"Could not find matching '{close_char}' for index {open_idx}.")


def _extract_tensor_data_expr(tensor_expr: str) -> str:
    """Extract the first list expression from tensor(...)."""
    list_start = tensor_expr.find("[")
    if list_start == -1:
        raise ValueError("No list expression found inside tensor(...) block.")
    list_end = _extract_balanced(tensor_expr, list_start, "[", "]")
    return tensor_expr[list_start : list_end + 1]


def parse_denoising_log(log_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Parse a log file into:
    - xyz_seq: shape (num_frames, horizon, 3)
    - frame_labels: e.g. ['initial', 'timestep 90', ...]
    """
    text = log_path.read_text(encoding="utf-8")
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        raise ValueError(f"No trajectory blocks found in: {log_path}")

    frames = []
    labels: List[str] = []

    for i, match in enumerate(matches):
        block_start = match.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[block_start:block_end]

        tensor_pos = block_text.find("tensor(")
        if tensor_pos == -1:
            raise ValueError(f"'tensor(' not found after header: {match.group(1)}")

        abs_tensor_start = block_start + tensor_pos
        open_paren_idx = abs_tensor_start + len("tensor")
        close_paren_idx = _extract_balanced(text, open_paren_idx, "(", ")")
        tensor_expr = text[open_paren_idx + 1 : close_paren_idx]
        list_expr = _extract_tensor_data_expr(tensor_expr)

        try:
            arr = np.asarray(ast.literal_eval(list_expr), dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"Failed to parse tensor data near header: {match.group(1)}") from exc

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(
                f"Unexpected trajectory shape {arr.shape}; expected (horizon, action_dim>=3)."
            )

        frames.append(arr[:, :3])
        if match.group(2) is None:
            labels.append("initial")
        else:
            labels.append(f"timestep {match.group(2)}")

    return np.stack(frames, axis=0), labels


def _compute_center_radius(points_xyz: np.ndarray, pad_ratio: float = 0.12) -> Tuple[np.ndarray, float]:
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() * 0.5
    radius = max(radius * (1.0 + pad_ratio), 1e-3)
    return center, float(radius)


def _set_equal_3d_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _apply_preferred_style() -> None:
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue


def _build_branch_cmap(name: str, colors: Sequence[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, list(colors))


def _compute_step_displacement(xyz_seq: np.ndarray) -> np.ndarray:
    displacement = np.zeros(xyz_seq.shape[0], dtype=np.float64)
    if xyz_seq.shape[0] > 1:
        displacement[1:] = np.mean(
            np.linalg.norm(xyz_seq[1:] - xyz_seq[:-1], axis=-1),
            axis=1,
        )
    return displacement


def _find_branch_start(branch_gap: np.ndarray, threshold: float) -> int | None:
    nonzero_idx = np.flatnonzero(branch_gap > threshold)
    if nonzero_idx.size == 0:
        return None
    return int(nonzero_idx[0])


def build_comparison_data(
    original_xyz: np.ndarray,
    original_labels: List[str],
    changed_xyz: np.ndarray,
    changed_labels: List[str],
    branch_threshold: float = 1e-9,
) -> ComparisonData:
    if original_xyz.shape != changed_xyz.shape:
        raise ValueError(
            f"Shape mismatch between logs: {original_xyz.shape} vs {changed_xyz.shape}"
        )
    if original_labels != changed_labels:
        raise ValueError("Frame labels do not match between the two logs.")

    branch_gap = np.mean(np.linalg.norm(original_xyz - changed_xyz, axis=-1), axis=1)
    branch_start_idx = _find_branch_start(branch_gap, branch_threshold)

    return ComparisonData(
        original_xyz=original_xyz,
        changed_xyz=changed_xyz,
        labels=original_labels,
        original_step_disp=_compute_step_displacement(original_xyz),
        changed_step_disp=_compute_step_displacement(changed_xyz),
        branch_gap=branch_gap,
        branch_start_idx=branch_start_idx,
    )


def _short_label(label: str) -> str:
    if label == "initial":
        return "init"
    match = re.search(r"(-?\d+)$", label)
    return match.group(1) if match else label


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return safe or "frame"


def _compute_focus_points(xyz_seq: np.ndarray, tail_frames: int = 4) -> np.ndarray:
    """
    Focus the camera on the converged region so the final trajectory is large enough
    to inspect. The last few denoising frames are included to preserve local motion.
    """
    tail_frames = max(1, min(int(tail_frames), xyz_seq.shape[0]))
    return xyz_seq[-tail_frames:].reshape(-1, 3)


def _setup_branch_axis(
    ax,
    xyz_seq: np.ndarray,
    title: str,
    final_marker: str,
    final_color: str,
    final_label: str,
) -> object:
    focus_points = _compute_focus_points(xyz_seq, tail_frames=4)
    center, radius = _compute_center_radius(focus_points, pad_ratio=0.10)
    radius = max(radius, 0.03)
    initial = xyz_seq[0]
    final = xyz_seq[-1]

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)
    ax.view_init(elev=24, azim=38)
    _set_equal_3d_limits(ax, center, radius)

    ax.scatter(
        initial[:, 0],
        initial[:, 1],
        initial[:, 2],
        marker="x",
        s=38,
        color="#9AA0A6",
        alpha=0.72,
        label="Initial noise",
    )
    ax.scatter(
        final[:, 0],
        final[:, 1],
        final[:, 2],
        marker=final_marker,
        s=48,
        facecolors="none",
        edgecolors=final_color,
        linewidths=0.9,
        alpha=0.9,
        label=final_label,
    )

    info_text = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.1,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#CCCCCC"),
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.02), fontsize=9, framealpha=0.92)
    return info_text


def _setup_comparison_figure(compare: ComparisonData) -> ComparisonRenderState:
    _apply_preferred_style()

    num_frames, horizon, _ = compare.original_xyz.shape
    x_idx = np.arange(num_frames)
    tick_labels = [_short_label(label) for label in compare.labels]

    fig = plt.figure(figsize=(17.2, 9.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[2.2, 1.0],
        left=0.035,
        right=0.985,
        bottom=0.075,
        top=0.93,
        wspace=0.08,
        hspace=0.22,
    )
    ax3d_original = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d_changed = fig.add_subplot(gs[0, 1], projection="3d")
    ax_gap = fig.add_subplot(gs[1, 0])
    ax_move = fig.add_subplot(gs[1, 1])

    original_cmap = _build_branch_cmap(
        "original_time",
        ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#0D47A1"],
    )
    changed_cmap = _build_branch_cmap(
        "changed_time",
        ["#FFF3E0", "#FFCC80", "#FFA726", "#F4511E", "#BF360C"],
    )
    original_colors = original_cmap(np.linspace(0.08, 0.95, horizon))
    changed_colors = changed_cmap(np.linspace(0.08, 0.95, horizon))

    original_info_text = _setup_branch_axis(
        ax=ax3d_original,
        xyz_seq=compare.original_xyz,
        title="Original Condition",
        final_marker="o",
        final_color=ORIGINAL_BASE_COLOR,
        final_label="Original final",
    )
    changed_info_text = _setup_branch_axis(
        ax=ax3d_changed,
        xyz_seq=compare.changed_xyz,
        title="Changed Condition",
        final_marker="^",
        final_color=CHANGED_BASE_COLOR,
        final_label="Changed final",
    )

    original_lines = []
    changed_lines = []
    for h in range(horizon):
        (original_line,) = ax3d_original.plot(
            [],
            [],
            [],
            color=original_colors[h],
            linewidth=1.35,
            alpha=0.72,
        )
        (changed_line,) = ax3d_changed.plot(
            [],
            [],
            [],
            color=changed_colors[h],
            linewidth=1.35,
            alpha=0.72,
        )
        original_lines.append(original_line)
        changed_lines.append(changed_line)

    initial = compare.original_xyz[0]
    original_scatter = ax3d_original.scatter(
        initial[:, 0],
        initial[:, 1],
        initial[:, 2],
        c=np.arange(horizon),
        cmap=original_cmap,
        vmin=0,
        vmax=horizon - 1,
        s=84,
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        depthshade=True,
        label="Original current",
    )
    changed_scatter = ax3d_changed.scatter(
        initial[:, 0],
        initial[:, 1],
        initial[:, 2],
        c=np.arange(horizon),
        cmap=changed_cmap,
        vmin=0,
        vmax=horizon - 1,
        s=88,
        marker="^",
        edgecolors="white",
        linewidths=0.5,
        depthshade=True,
        label="Changed current",
    )

    ax_gap.set_title("Branch Gap", fontsize=12, fontweight="bold")
    ax_gap.set_ylabel("Mean |original - changed|")
    ax_gap.plot(x_idx, compare.branch_gap, color=GAP_BASE_COLOR, linewidth=2.2)
    ax_gap.scatter(x_idx, compare.branch_gap, color=GAP_BASE_COLOR, s=20, alpha=0.9)
    if compare.branch_start_idx is not None:
        start = compare.branch_start_idx
        ax_gap.axvspan(start - 0.5, num_frames - 0.5, color="#FDE68A", alpha=0.28)
        ax_gap.axvline(
            start,
            color=BRANCH_HIGHLIGHT_COLOR,
            linestyle="--",
            linewidth=1.6,
            alpha=0.95,
        )
        branch_note = f"first divergence: {compare.labels[start]}"
    else:
        branch_note = "no divergence detected"
    ax_gap.text(
        0.02,
        0.97,
        branch_note,
        transform=ax_gap.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="#DDDDDD"),
    )
    gap_cursor = ax_gap.axvline(0, color="#111827", linewidth=2.0, alpha=0.95)
    gap_cursor_dot = ax_gap.scatter([0], [compare.branch_gap[0]], color="#111827", s=68, zorder=5)
    ax_gap.set_xlim(-0.2, num_frames - 0.8)
    gap_ymax = max(float(compare.branch_gap.max()), 1e-6)
    ax_gap.set_ylim(-0.03 * gap_ymax, gap_ymax * 1.12)
    ax_gap.set_xticks(x_idx)
    ax_gap.set_xticklabels(tick_labels, fontsize=9)

    ax_move.set_title("Per-Frame Denoising Movement", fontsize=12, fontweight="bold")
    ax_move.set_xlabel("Log Frame Label (init, 90 -> 0)")
    ax_move.set_ylabel("Mean |delta xyz|")
    ax_move.plot(x_idx, compare.original_step_disp, color=ORIGINAL_BASE_COLOR, linewidth=2.0, label="Original")
    ax_move.scatter(x_idx, compare.original_step_disp, color=ORIGINAL_BASE_COLOR, s=20, alpha=0.88)
    ax_move.plot(
        x_idx,
        compare.changed_step_disp,
        color=CHANGED_BASE_COLOR,
        linewidth=2.0,
        label="Changed condition",
    )
    ax_move.scatter(x_idx, compare.changed_step_disp, color=CHANGED_BASE_COLOR, s=20, alpha=0.88)
    if compare.branch_start_idx is not None:
        ax_move.axvline(
            compare.branch_start_idx,
            color=BRANCH_HIGHLIGHT_COLOR,
            linestyle="--",
            linewidth=1.6,
            alpha=0.95,
        )
    move_cursor = ax_move.axvline(0, color="#111827", linewidth=2.0, alpha=0.95)
    original_move_dot = ax_move.scatter(
        [0],
        [compare.original_step_disp[0]],
        color=ORIGINAL_BASE_COLOR,
        s=66,
        zorder=5,
    )
    changed_move_dot = ax_move.scatter(
        [0],
        [compare.changed_step_disp[0]],
        color=CHANGED_BASE_COLOR,
        s=66,
        zorder=5,
    )
    ax_move.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax_move.set_xlim(-0.2, num_frames - 0.8)
    move_ymax = max(
        float(compare.original_step_disp.max()),
        float(compare.changed_step_disp.max()),
        1e-6,
    )
    ax_move.set_ylim(-0.03 * move_ymax, move_ymax * 1.12)
    ax_move.set_xticks(x_idx)
    ax_move.set_xticklabels(tick_labels, fontsize=9)

    return ComparisonRenderState(
        fig=fig,
        ax3d_original=ax3d_original,
        ax3d_changed=ax3d_changed,
        ax_gap=ax_gap,
        ax_move=ax_move,
        original_lines=original_lines,
        changed_lines=changed_lines,
        original_scatter=original_scatter,
        changed_scatter=changed_scatter,
        original_info_text=original_info_text,
        changed_info_text=changed_info_text,
        gap_cursor=gap_cursor,
        gap_cursor_dot=gap_cursor_dot,
        move_cursor=move_cursor,
        original_move_dot=original_move_dot,
        changed_move_dot=changed_move_dot,
    )


def _update_comparison_state(
    state: ComparisonRenderState,
    compare: ComparisonData,
    frame_idx: int,
    trail: int,
) -> List[object]:
    trail = max(trail, 1)
    trail_start = max(0, frame_idx - trail + 1)

    original_pts = compare.original_xyz[frame_idx]
    changed_pts = compare.changed_xyz[frame_idx]

    state.original_scatter._offsets3d = (
        original_pts[:, 0],
        original_pts[:, 1],
        original_pts[:, 2],
    )
    state.changed_scatter._offsets3d = (
        changed_pts[:, 0],
        changed_pts[:, 1],
        changed_pts[:, 2],
    )

    for horizon_idx, line in enumerate(state.original_lines):
        segment = compare.original_xyz[trail_start : frame_idx + 1, horizon_idx, :]
        line.set_data(segment[:, 0], segment[:, 1])
        line.set_3d_properties(segment[:, 2])

    for horizon_idx, line in enumerate(state.changed_lines):
        segment = compare.changed_xyz[trail_start : frame_idx + 1, horizon_idx, :]
        line.set_data(segment[:, 0], segment[:, 1])
        line.set_3d_properties(segment[:, 2])

    if compare.branch_start_idx is None or frame_idx < compare.branch_start_idx:
        branch_status = "shared path"
    else:
        branch_status = f"branched since {compare.labels[compare.branch_start_idx]}"

    state.original_info_text.set_text(
        f"{compare.labels[frame_idx]}  ({frame_idx + 1}/{compare.original_xyz.shape[0]})\n"
        f"original step move: {compare.original_step_disp[frame_idx]:.4f}\n"
        f"branch gap: {compare.branch_gap[frame_idx]:.4f}\n"
        f"status: {branch_status}"
    )
    state.changed_info_text.set_text(
        f"{compare.labels[frame_idx]}  ({frame_idx + 1}/{compare.changed_xyz.shape[0]})\n"
        f"changed step move: {compare.changed_step_disp[frame_idx]:.4f}\n"
        f"branch gap: {compare.branch_gap[frame_idx]:.4f}\n"
        f"status: {branch_status}"
    )

    state.gap_cursor.set_xdata([frame_idx, frame_idx])
    state.gap_cursor_dot.set_offsets(np.array([[frame_idx, compare.branch_gap[frame_idx]]]))

    state.move_cursor.set_xdata([frame_idx, frame_idx])
    state.original_move_dot.set_offsets(
        np.array([[frame_idx, compare.original_step_disp[frame_idx]]])
    )
    state.changed_move_dot.set_offsets(
        np.array([[frame_idx, compare.changed_step_disp[frame_idx]]])
    )

    return [
        state.original_scatter,
        state.changed_scatter,
        state.original_info_text,
        state.changed_info_text,
        state.gap_cursor,
        state.gap_cursor_dot,
        state.move_cursor,
        state.original_move_dot,
        state.changed_move_dot,
        *state.original_lines,
        *state.changed_lines,
    ]


def create_comparison_animation(
    compare: ComparisonData,
    output_path: Path,
    fps: int = 4,
    trail: int = 5,
    dpi: int = 140,
) -> Path:
    if compare.original_xyz.shape[0] < 2:
        raise ValueError("Need at least 2 frames to animate denoising.")

    state = _setup_comparison_figure(compare)

    animation = FuncAnimation(
        fig=state.fig,
        func=lambda idx: _update_comparison_state(state, compare, idx, trail),
        frames=compare.original_xyz.shape[0],
        interval=int(1000 / max(fps, 1)),
        blit=False,
        repeat=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_ext = output_path.suffix.lower()

    if output_ext == ".gif":
        writer = PillowWriter(fps=fps)
        animation.save(str(output_path), writer=writer, dpi=dpi)
    else:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            animation.save(str(output_path), writer=writer, dpi=dpi)
        except Exception as exc:
            fallback = output_path.with_suffix(".gif")
            print(
                f"[warn] Failed to save '{output_path.name}' as video ({exc}). "
                f"Saving GIF instead: '{fallback.name}'"
            )
            writer = PillowWriter(fps=fps)
            animation.save(str(fallback), writer=writer, dpi=dpi)
            output_path = fallback

    plt.close(state.fig)
    return output_path


def save_comparison_frame_sequence(
    compare: ComparisonData,
    output_dir: Path,
    trail: int = 5,
    dpi: int = 170,
) -> Path:
    num_frames = compare.original_xyz.shape[0]
    if num_frames < 1:
        raise ValueError("No frames found to export.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = ["frame_idx\tlabel\tbranch_gap\toriginal_step_move\tchanged_step_move\tfile_name"]

    for frame_idx in range(num_frames):
        state = _setup_comparison_figure(compare)
        _update_comparison_state(state, compare, frame_idx, trail)

        filename = f"frame_{frame_idx:02d}_{_safe_name(compare.labels[frame_idx])}.png"
        save_path = output_dir / filename
        state.fig.savefig(save_path, dpi=dpi)
        plt.close(state.fig)

        frame_rows.append(
            "\t".join(
                [
                    str(frame_idx),
                    compare.labels[frame_idx],
                    f"{compare.branch_gap[frame_idx]:.6f}",
                    f"{compare.original_step_disp[frame_idx]:.6f}",
                    f"{compare.changed_step_disp[frame_idx]:.6f}",
                    filename,
                ]
            )
        )

    (output_dir / "index.tsv").write_text("\n".join(frame_rows) + "\n", encoding="utf-8")
    return output_dir


def _resolve_inputs(
    input_dir: Path,
    traj_idx: int,
    change_suffix: str,
    original_input: Path | None,
    changed_input: Path | None,
) -> Tuple[Path, Path]:
    if original_input is None:
        original_input = input_dir / f"action_denoising_traj_{traj_idx}.txt"
    if changed_input is None:
        changed_input = input_dir / f"action_denoising_traj_{traj_idx}_{change_suffix}.txt"
    return original_input, changed_input


def build_arg_parser(default_input_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two 3D diffusion action denoising logs (xyz only)."
    )
    parser.add_argument(
        "--traj-idx",
        type=int,
        default=0,
        help="Trajectory index i for action_denoising_traj_{i}.txt (default: 0)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Directory that contains the denoising logs (default: {default_input_dir})",
    )
    parser.add_argument(
        "--original-input",
        type=Path,
        default=None,
        help="Optional explicit path to the original log file.",
    )
    parser.add_argument(
        "--changed-input",
        type=Path,
        default=None,
        help="Optional explicit path to the changed-condition log file.",
    )
    parser.add_argument(
        "--change-suffix",
        type=str,
        default="change_obs_30",
        help="Suffix used for the changed-condition file name (default: change_obs_30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output animation path (.gif or .mp4). If omitted, a default name is used.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="If set, export each comparison frame as PNG files in this directory.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip GIF/MP4 animation export and only save per-step PNG frames.",
    )
    parser.add_argument("--fps", type=int, default=4, help="Animation FPS (default: 4)")
    parser.add_argument(
        "--trail",
        type=int,
        default=5,
        help="History length shown per horizon point (default: 5 frames)",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.add_argument(
        "--branch-threshold",
        type=float,
        default=1e-9,
        help="Numerical threshold used to detect the first divergence frame.",
    )
    return parser


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = build_arg_parser(script_dir)
    args = parser.parse_args()

    original_input, changed_input = _resolve_inputs(
        input_dir=args.input_dir,
        traj_idx=args.traj_idx,
        change_suffix=args.change_suffix,
        original_input=args.original_input,
        changed_input=args.changed_input,
    )

    if args.output is None:
        output_path = args.input_dir / (
            f"action_denoising_traj_{args.traj_idx}_compare_{args.change_suffix}.gif"
        )
    else:
        output_path = args.output

    if not original_input.exists():
        raise FileNotFoundError(f"Original input log not found: {original_input}")
    if not changed_input.exists():
        raise FileNotFoundError(f"Changed-condition input log not found: {changed_input}")

    original_xyz, original_labels = parse_denoising_log(original_input)
    changed_xyz, changed_labels = parse_denoising_log(changed_input)
    compare = build_comparison_data(
        original_xyz=original_xyz,
        original_labels=original_labels,
        changed_xyz=changed_xyz,
        changed_labels=changed_labels,
        branch_threshold=max(args.branch_threshold, 0.0),
    )

    print(
        "[info] Parsed comparison: "
        f"frames={compare.original_xyz.shape[0]}, "
        f"horizon={compare.original_xyz.shape[1]}, "
        f"xyz_dim=3"
    )
    print(f"[info] Original log: {original_input}")
    print(f"[info] Changed log:  {changed_input}")
    if compare.branch_start_idx is None:
        print("[info] No divergence detected between the two runs.")
    else:
        print(
            "[info] First divergence at "
            f"frame {compare.branch_start_idx} ({compare.labels[compare.branch_start_idx]}), "
            f"mean gap={compare.branch_gap[compare.branch_start_idx]:.6f}"
        )

    did_work = False

    if not args.skip_animation:
        saved_path = create_comparison_animation(
            compare=compare,
            output_path=output_path,
            fps=max(args.fps, 1),
            trail=max(args.trail, 1),
            dpi=max(args.dpi, 72),
        )
        print(f"[done] Saved comparison animation: {saved_path}")
        did_work = True

    if args.frames_dir is not None:
        frames_path = save_comparison_frame_sequence(
            compare=compare,
            output_dir=args.frames_dir,
            trail=max(args.trail, 1),
            dpi=max(args.dpi, 72),
        )
        print(f"[done] Saved comparison frame sequence: {frames_path}")
        did_work = True

    if not did_work:
        raise ValueError("Nothing to export. Use animation output or provide --frames-dir.")


if __name__ == "__main__":
    main()
