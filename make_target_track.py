from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


BOX_FIELDS = [
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "bottom_center_x",
    "bottom_center_y",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "x1_norm",
    "y1_norm",
    "x2_norm",
    "y2_norm",
    "bottom_center_x_norm",
    "bottom_center_y_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one canonical cut-in target track and interpolate only short gaps."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/target_gray"))
    parser.add_argument("--start", type=float, default=6.0)
    parser.add_argument("--end", type=float, default=17.0)
    parser.add_argument("--max-gap", type=int, default=4)
    return parser.parse_args()


def load_target_rows(path: Path, track_id: int) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = [row for row in csv.DictReader(csv_file) if int(row["track_id"]) == track_id]
    if not rows:
        raise ValueError(f"Track ID {track_id} was not found in {path}")

    parsed = []
    for row in rows:
        parsed_row = dict(row)
        parsed_row["frame_idx"] = int(row["frame_idx"])
        parsed_row["frame_number"] = int(row["frame_number"])
        parsed_row["clip_frame_idx"] = int(row["clip_frame_idx"])
        parsed_row["source_track_id"] = int(row["track_id"])
        parsed_row["confidence"] = float(row["confidence"])
        parsed_row["timestamp_s"] = float(row["timestamp_s"])
        parsed_row["clip_time_s"] = float(row["clip_time_s"])
        for field in BOX_FIELDS:
            parsed_row[field] = float(row[field])
        parsed_row["canonical_track_id"] = "cutin_gray_01"
        parsed_row["is_interpolated"] = False
        parsed.append(parsed_row)
    return sorted(parsed, key=lambda row: row["frame_idx"])


def interpolate_short_gaps(rows: list[dict], fps: float, max_gap: int) -> list[dict]:
    completed: list[dict] = []
    for current, following in zip(rows, rows[1:]):
        completed.append(current)
        missing = following["frame_idx"] - current["frame_idx"] - 1
        if missing <= 0 or missing > max_gap:
            continue
        for offset in range(1, missing + 1):
            ratio = offset / (missing + 1)
            interpolated = dict(current)
            frame_idx = current["frame_idx"] + offset
            interpolated["frame_idx"] = frame_idx
            interpolated["frame_number"] = frame_idx + 1
            interpolated["clip_frame_idx"] = current["clip_frame_idx"] + offset
            interpolated["timestamp_s"] = frame_idx / fps
            interpolated["clip_time_s"] = interpolated["clip_frame_idx"] / fps
            interpolated["confidence"] = min(current["confidence"], following["confidence"])
            interpolated["class_id"] = -1
            interpolated["class_name"] = "interpolated_vehicle"
            interpolated["is_interpolated"] = True
            for field in BOX_FIELDS:
                interpolated[field] = current[field] + ratio * (following[field] - current[field])
            completed.append(interpolated)
    completed.append(rows[-1])
    return sorted(completed, key=lambda row: row["frame_idx"])


def write_target_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "canonical_track_id",
        "source_track_id",
        "frame_idx",
        "frame_number",
        "clip_frame_idx",
        "timestamp_s",
        "clip_time_s",
        "is_interpolated",
        "confidence",
        "class_id",
        "class_name",
        "object_group",
        *BOX_FIELDS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_target_video(
    source: Path,
    output: Path,
    rows: list[dict],
    start_s: float,
    end_s: float,
    fps: float,
) -> int:
    rows_by_frame = {row["frame_idx"]: row for row in rows}
    cap = cv2.VideoCapture(str(source))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, round(start_s * fps))
    end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create {output}")

    written = 0
    try:
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            row = rows_by_frame.get(frame_idx)
            if row is not None:
                color = (0, 200, 255) if not row["is_interpolated"] else (0, 140, 255)
                point1 = (round(row["x1"]), round(row["y1"]))
                point2 = (round(row["x2"]), round(row["y2"]))
                cv2.rectangle(frame, point1, point2, color, 3)
                suffix = " (interpolated)" if row["is_interpolated"] else ""
                cv2.putText(
                    frame,
                    f"TARGET cutin_gray_01{suffix}",
                    (point1[0], max(26, point1[1] - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame,
                    (round(row["bottom_center_x"]), round(row["bottom_center_y"])),
                    5,
                    color,
                    -1,
                )
            cv2.putText(
                frame,
                f"source time {frame_idx / fps:05.2f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()
    return written


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    tracks = args.tracks.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not tracks.is_file():
        raise FileNotFoundError(tracks)

    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    observed = load_target_rows(tracks, args.track_id)
    completed = interpolate_short_gaps(observed, fps, args.max_gap)
    csv_path = output / "target_track.csv"
    video_path = output / "target_only.mp4"
    write_target_csv(csv_path, completed)
    video_frames = draw_target_video(source, video_path, completed, args.start, args.end, fps)

    interpolated_count = sum(bool(row["is_interpolated"]) for row in completed)
    print(f"Observed target frames: {len(observed)}")
    print(f"Interpolated short-gap frames: {interpolated_count}")
    print(f"Final target rows: {len(completed)}")
    print(f"Output video frames: {video_frames}")
    print(f"CSV: {csv_path}")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
