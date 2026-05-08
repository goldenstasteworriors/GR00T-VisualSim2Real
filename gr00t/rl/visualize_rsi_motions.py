import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _root_xy(data, key):
    return data[key].item()["body_positions"][:, 0, :2]


def _load_motion(path):
    data = np.load(path, allow_pickle=True)
    return {
        "robot": _root_xy(data, "robot"),
        "front_object": _root_xy(data, "bottle_table"),
        "hold_object": _root_xy(data, "bottle_hand"),
        "tray": _root_xy(data, "tray_table"),
        "fps": float(data["robot"].item()["fps"]),
        "num_frames": int(data["robot"].item()["body_positions"].shape[0]),
    }


def _axis_limits(motion):
    pts = np.concatenate(
        [motion["robot"], motion["front_object"], motion["hold_object"], motion["tray"]], axis=0
    )
    lo = pts.min(axis=0) - 0.5
    hi = pts.max(axis=0) + 0.5
    return lo, hi


def _project(points, lo, hi, width, height, margin):
    scale = min((width - 2 * margin) / (hi[0] - lo[0]), (height - 2 * margin) / (hi[1] - lo[1]))
    x = margin + (points[..., 0] - lo[0]) * scale
    y = height - margin - (points[..., 1] - lo[1]) * scale
    return np.stack([x, y], axis=-1)


def _draw_polyline(draw, points, color, width=2):
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)


def _draw_marker(draw, point, color, radius=7, square=False):
    x, y = map(float, point)
    if square:
        draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=color)
    else:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def render_frame(motion, frame_idx, title, width=720, height=720):
    margin = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    lo, hi = _axis_limits(motion)

    # Grid.
    for i in range(6):
        x = margin + i * (width - 2 * margin) / 5
        y = margin + i * (height - 2 * margin) / 5
        draw.line([(x, margin), (x, height - margin)], fill=(225, 225, 225), width=1)
        draw.line([(margin, y), (width - margin, y)], fill=(225, 225, 225), width=1)
    draw.rectangle([margin, margin, width - margin, height - margin], outline=(180, 180, 180))

    colors = {
        "robot": (31, 119, 180),
        "front_object": (255, 127, 14),
        "hold_object": (44, 160, 44),
        "tray": (214, 39, 40),
    }
    for key, color in colors.items():
        pts = _project(motion[key], lo, hi, width, height, margin)
        _draw_polyline(draw, pts, color, width=2)
        _draw_marker(draw, pts[frame_idx], color, radius=8, square=(key == "tray"))

    draw.text((20, 18), title, fill=(20, 20, 20))
    legend_x, legend_y = width - 210, 25
    for i, (key, color) in enumerate(colors.items()):
        y = legend_y + i * 24
        _draw_marker(draw, (legend_x, y + 8), color, radius=6, square=(key == "tray"))
        draw.text((legend_x + 18, y), key, fill=(20, 20, 20))
    draw.text((20, height - 35), "top-down XY; trajectories are full demo, markers are current frame", fill=(80, 80, 80))
    return image


def save_contact_sheet(motion, output_path, motion_name, num_frames):
    sample_ids = np.linspace(0, motion["num_frames"] - 1, num_frames, dtype=int)
    cols = min(5, num_frames)
    rows = int(np.ceil(num_frames / cols))
    tile_w, tile_h = 720, 720
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for tile_idx, frame_idx in enumerate(sample_ids):
        t = frame_idx / motion["fps"]
        tile = render_frame(motion, int(frame_idx), f"{motion_name}\nframe={frame_idx}, t={t:.2f}s", tile_w, tile_h)
        sheet.paste(tile, ((tile_idx % cols) * tile_w, (tile_idx // cols) * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_gif(motion, output_path, motion_name, stride):
    frames = []
    for frame_idx in range(0, motion["num_frames"], stride):
        t = frame_idx / motion["fps"]
        frame = render_frame(motion, frame_idx, f"{motion_name}  frame={frame_idx}, t={t:.2f}s")
        frames.append(np.asarray(frame))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=stride / motion["fps"], loop=0)


def write_summary(motion_files, output_path, num_per_sample, sample_interval_s):
    duration_per_sample = num_per_sample * sample_interval_s
    rows = []
    total_segments = 0
    for path in motion_files:
        data = np.load(path, allow_pickle=True)
        robot = data["robot"].item()
        fps = float(robot["fps"])
        num_frames = int(robot["dof_positions"].shape[0])
        duration = (num_frames - 1) / fps
        if duration_per_sample * 2 > duration:
            num_segments = 0
        else:
            num_segments = int((duration - duration_per_sample) / duration_per_sample)
        total_segments += num_segments
        rows.append(
            {
                "file": str(path),
                "fps": fps,
                "num_frames": num_frames,
                "duration_s": duration,
                "rsi_segments": num_segments,
                "rsi_candidate_frames": num_segments * num_per_sample,
            }
        )
    summary = {
        "motion_file_count": len(rows),
        "num_per_sample": num_per_sample,
        "sample_interval_s": sample_interval_s,
        "rsi_segment_count": total_segments,
        "rsi_candidate_frame_count": total_segments * num_per_sample,
        "motions": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Visualize RSI teleop motion snapshots.")
    parser.add_argument(
        "--motion-dir",
        default="gr00t/rl/data/motions/g1_wsdpt/33demos_675_775",
        help="Directory containing teleop .npz files used by reset_from_dataset.",
    )
    parser.add_argument("--motion-index", type=int, default=0, help="Which motion file to render.")
    parser.add_argument("--output-dir", default="logs_eval/rsi_motion_vis")
    parser.add_argument("--num-frames", type=int, default=15, help="Frames in contact sheet.")
    parser.add_argument("--gif", action="store_true", help="Also save an animated top-down GIF.")
    parser.add_argument("--gif-stride", type=int, default=10)
    parser.add_argument("--num-per-sample", type=int, default=10)
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir)
    motion_files = sorted(motion_dir.glob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No .npz files found under {motion_dir}")
    if args.motion_index < 0 or args.motion_index >= len(motion_files):
        raise IndexError(f"--motion-index must be in [0, {len(motion_files) - 1}]")

    output_dir = Path(args.output_dir)
    write_summary(
        motion_files,
        output_dir / "rsi_motion_summary.json",
        args.num_per_sample,
        args.sample_interval_s,
    )

    motion_path = motion_files[args.motion_index]
    motion = _load_motion(motion_path)
    stem = motion_path.stem
    save_contact_sheet(motion, output_dir / f"{stem}_contact_sheet.png", stem, args.num_frames)
    if args.gif:
        save_gif(motion, output_dir / f"{stem}_topdown.gif", stem, args.gif_stride)
    print(f"Saved RSI visualization to {output_dir}")


if __name__ == "__main__":
    main()
