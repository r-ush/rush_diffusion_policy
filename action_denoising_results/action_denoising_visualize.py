#!/usr/bin/env python3
"""
Visualize diffusion action denoising logs in 3D (x, y, z only).

Expected log format example:
    initial trajectory:
    tensor([[[...]]], device='cuda:0')
    timestep 90, trajectory:
    tensor([[[...]]], device='cuda:0')
    ...
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap


HEADER_RE = re.compile(
    r"^(initial trajectory:|timestep\s+(-?\d+),\s*trajectory:)\s*$",
    flags=re.MULTILINE,
)


def _extract_balanced(text: str, open_idx: int, open_char: str, close_char: str) -> int:
    """Return matching closing-char index for a balanced expression."""
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
    """
    From tensor(...) argument text, extract the first list expression: [[[...]]].
    """
    list_start = tensor_expr.find("[")
    if list_start == -1:
        raise ValueError("No list expression found inside tensor(...) block.")
    list_end = _extract_balanced(tensor_expr, list_start, "[", "]")
    return tensor_expr[list_start : list_end + 1]


def parse_denoising_log(log_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Parse log file into:
    - xyz_seq: shape (num_frames, horizon, 3)
    - frame_labels: readable label per frame (e.g. 'initial', 'timestep 90')
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

        frames.append(arr[:, :3])  # keep xyz only

        if match.group(2) is None:
            labels.append("initial")
        else:
            labels.append(f"timestep {match.group(2)}")

    xyz_seq = np.stack(frames, axis=0)
    return xyz_seq, labels


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


def _build_time_red_cmap() -> LinearSegmentedColormap:
    # Light-to-deep red gradient to represent horizon/time order.
    return LinearSegmentedColormap.from_list(
        "time_red",
        ["#FFE6E6", "#FFB3B3", "#FF6B6B", "#E53935", "#9C1C1C"],
    )


def create_animation(
    xyz_seq: np.ndarray,
    labels: List[str],
    output_path: Path,
    fps: int = 4,
    trail: int = 5,
    dpi: int = 140,
) -> Path:
    num_frames, horizon, _ = xyz_seq.shape
    if num_frames < 2:
        raise ValueError("Need at least 2 frames to animate denoising.")

    displacement = np.zeros(num_frames, dtype=np.float64)
    displacement[1:] = np.mean(
        np.linalg.norm(xyz_seq[1:] - xyz_seq[:-1], axis=-1),
        axis=1,
    )

    _apply_preferred_style()
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.2, 1.4],
        left=0.03,
        right=0.94,
        bottom=0.08,
        top=0.92,
        wspace=0.18,
    )
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    axm = fig.add_subplot(gs[0, 1])

    ax3d.set_title("Action Denoising In XYZ Space", fontsize=13, fontweight="bold", pad=12)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.grid(False)
    ax3d.view_init(elev=24, azim=38)
    initial = xyz_seq[0]
    final = xyz_seq[-1]
    init_center, init_radius = _compute_center_radius(initial, pad_ratio=0.05)
    _set_equal_3d_limits(ax3d, init_center, init_radius)
    ax3d.scatter(
        initial[:, 0],
        initial[:, 1],
        initial[:, 2],
        marker="x",
        s=34,
        color="#9AA0A6",
        alpha=0.65,
        label="Initial (noise)",
    )
    ax3d.scatter(
        final[:, 0],
        final[:, 1],
        final[:, 2],
        marker="o",
        s=48,
        facecolors="none",
        edgecolors="#1E5BD9",
        linewidths=0.65,
        alpha=0.9,
        label="Final",
    )

    red_cmap = _build_time_red_cmap()
    horizon_colors = red_cmap(np.linspace(0.08, 0.95, horizon))
    history_lines = []
    for h in range(horizon):
        (line,) = ax3d.plot([], [], [], color=horizon_colors[h], linewidth=1.2, alpha=0.45)
        history_lines.append(line)

    scatter = ax3d.scatter(
        initial[:, 0],
        initial[:, 1],
        initial[:, 2],
        label="Current Points",
        c=np.arange(horizon),
        cmap=red_cmap,
        vmin=0,
        vmax=horizon - 1,
        s=72,
        edgecolors="white",
        linewidths=0.5,
        depthshade=True,
    )

    prev_axes_grid = plt.rcParams.get("axes.grid", False)
    plt.rcParams["axes.grid"] = False
    cbar = fig.colorbar(scatter, ax=ax3d, pad=0.02, fraction=0.03)
    plt.rcParams["axes.grid"] = prev_axes_grid
    cbar.set_label("Time Index (Horizon)")

    info_text = ax3d.text2D(
        0.64,
        0.96,
        "",
        transform=ax3d.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )
    ax3d.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=9, framealpha=0.9)

    x_idx = np.arange(num_frames)
    axm.set_title("Denoising Step Movement", fontsize=12, fontweight="bold")
    axm.set_xlabel("Frame Index")
    axm.set_ylabel("Mean |delta xyz|")
    axm.plot(x_idx, displacement, color="#4C78A8", linewidth=2.0)
    axm.scatter(x_idx, displacement, color="#4C78A8", s=20, alpha=0.85)
    cursor = axm.axvline(0, color="#E45756", linewidth=2.0, alpha=0.9)
    cursor_dot = axm.scatter([0], [displacement[0]], color="#E45756", s=70, zorder=5)
    axm.set_xlim(-0.2, num_frames - 0.8)
    y_max = max(float(displacement.max()), 1e-6)
    axm.set_ylim(-0.03 * y_max, y_max * 1.08)

    def update(frame_idx: int):
        pts = xyz_seq[frame_idx]
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        start = 0
        for h, line in enumerate(history_lines):
            seg = xyz_seq[start : frame_idx + 1, h, :]
            line.set_data(seg[:, 0], seg[:, 1])
            line.set_3d_properties(seg[:, 2])

        info_text.set_text(
            f"{labels[frame_idx]}  ({frame_idx + 1}/{num_frames})\n"
            f"mean |delta xyz|: {displacement[frame_idx]:.4f}"
        )
        cursor.set_xdata([frame_idx, frame_idx])
        cursor_dot.set_offsets(np.array([[frame_idx, displacement[frame_idx]]]))
        return [scatter, *history_lines, info_text, cursor, cursor_dot]

    animation = FuncAnimation(
        fig=fig,
        func=update,
        frames=num_frames,
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

    plt.close(fig)
    return output_path


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return safe or "frame"


def save_frame_sequence(
    xyz_seq: np.ndarray,
    labels: List[str],
    output_dir: Path,
    trail: int = 5,
    dpi: int = 170,
) -> Path:
    num_frames, horizon, _ = xyz_seq.shape
    if num_frames < 1:
        raise ValueError("No frames found to export.")

    _apply_preferred_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    displacement = np.zeros(num_frames, dtype=np.float64)
    if num_frames > 1:
        displacement[1:] = np.mean(
            np.linalg.norm(xyz_seq[1:] - xyz_seq[:-1], axis=-1),
            axis=1,
        )

    frame_rows = ["frame_idx\tlabel\tfile_name"]
    red_cmap = _build_time_red_cmap()
    horizon_colors = red_cmap(np.linspace(0.08, 0.95, horizon))
    initial = xyz_seq[0]
    final = xyz_seq[-1]
    init_center, init_radius = _compute_center_radius(initial, pad_ratio=0.05)

    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[3.2, 1.4],
            left=-0.2,
            right=0.94,
            bottom=0.08,
            top=0.92,
            wspace=0.18,
        )
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        axm = fig.add_subplot(gs[0, 1])

        ax3d.set_title("Action Denoising In XYZ Space", fontsize=13, fontweight="bold", pad=12)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.grid(False)
        ax3d.view_init(elev=24, azim=38)
        _set_equal_3d_limits(ax3d, init_center, init_radius)

        ax3d.scatter(
            initial[:, 0],
            initial[:, 1],
            initial[:, 2],
            marker="x",
            s=34,
            color="#9AA0A6",
            alpha=0.65,
            label="Initial (noise)",
        )
        ax3d.scatter(
            final[:, 0],
            final[:, 1],
            final[:, 2],
            marker="o",
            s=48,
            facecolors="none",
            edgecolors="#1E5BD9",
            linewidths=0.65,
            alpha=0.9,
            label="Final",
        )

        start = 0
        for h in range(horizon):
            seg = xyz_seq[start : frame_idx + 1, h, :]
            ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=horizon_colors[h], linewidth=1.2, alpha=0.55)

        pts = xyz_seq[frame_idx]
        scatter = ax3d.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            label="Current Points",
            c=np.arange(horizon),
            cmap=red_cmap,
            vmin=0,
            vmax=horizon - 1,
            s=72,
            edgecolors="white",
            linewidths=0.5,
            depthshade=True,
        )

        prev_axes_grid = plt.rcParams.get("axes.grid", False)
        plt.rcParams["axes.grid"] = False
        cbar = fig.colorbar(scatter, ax=ax3d, pad=0.02, fraction=0.03)
        plt.rcParams["axes.grid"] = prev_axes_grid
        cbar.set_label("Time Index (Horizon)")

        ax3d.text2D(
            0.64,
            0.96,
            f"{labels[frame_idx]}  ({frame_idx + 1}/{num_frames})\n"
            f"mean |delta xyz|: {displacement[frame_idx]:.4f}",
            transform=ax3d.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
        )
        ax3d.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=9, framealpha=0.9)

        x_idx = np.arange(num_frames)
        axm.set_title("Denoising Step Movement", fontsize=12, fontweight="bold")
        axm.set_xlabel("Frame Index")
        axm.set_ylabel("Mean |delta xyz|")
        axm.plot(x_idx, displacement, color="#4C78A8", linewidth=2.0)
        axm.scatter(x_idx, displacement, color="#4C78A8", s=20, alpha=0.85)
        axm.axvline(frame_idx, color="#E45756", linewidth=2.0, alpha=0.9)
        axm.scatter([frame_idx], [displacement[frame_idx]], color="#E45756", s=70, zorder=5)
        axm.set_xlim(-0.2, num_frames - 0.8)
        y_max = max(float(displacement.max()), 1e-6)
        axm.set_ylim(-0.03 * y_max, y_max * 1.08)

        filename = f"frame_{frame_idx:02d}_{_safe_name(labels[frame_idx])}.png"
        save_path = output_dir / filename
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)

        frame_rows.append(f"{frame_idx}\t{labels[frame_idx]}\t{filename}")

    (output_dir / "index.tsv").write_text("\n".join(frame_rows) + "\n", encoding="utf-8")
    return output_dir


def build_arg_parser(default_input: Path, default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D visualization of diffusion action denoising (x, y, z only)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Path to denoising log txt file (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output animation path (.gif or .mp4, default: {default_output})",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="If set, export each denoising step as PNG files in this directory.",
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
    return parser


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "log_action_0.txt"
    default_output = script_dir / "log_action_0_denoising_3d.gif"

    parser = build_arg_parser(default_input, default_output)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input log not found: {args.input}")

    xyz_seq, labels = parse_denoising_log(args.input)
    print(f"[info] Parsed {xyz_seq.shape[0]} frames, horizon={xyz_seq.shape[1]}, xyz_dim=3")

    did_work = False

    if not args.skip_animation:
        saved_path = create_animation(
            xyz_seq=xyz_seq,
            labels=labels,
            output_path=args.output,
            fps=args.fps,
            trail=max(args.trail, 1),
            dpi=max(args.dpi, 72),
        )
        print(f"[done] Saved animation: {saved_path}")
        did_work = True

    if args.frames_dir is not None:
        frames_path = save_frame_sequence(
            xyz_seq=xyz_seq,
            labels=labels,
            output_dir=args.frames_dir,
            trail=max(args.trail, 1),
            dpi=max(args.dpi, 72),
        )
        print(f"[done] Saved frame sequence: {frames_path}")
        did_work = True

    if not did_work:
        raise ValueError("Nothing to export. Use animation output or provide --frames-dir.")


if __name__ == "__main__":
    main()
