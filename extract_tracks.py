from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


VEHICLE_CLASS_IDS = [2, 3, 5, 7]


@dataclass
class Detection:
    run_id: str
    video_id: str
    source_video: str
    fps: float
    frame_width: int
    frame_height: int
    frame_idx: int
    frame_number: int
    clip_frame_idx: int
    timestamp_s: float
    clip_time_s: float
    track_id: int
    class_id: int
    class_name: str
    object_group: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    bottom_center_x: float
    bottom_center_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float
    bottom_center_x_norm: float
    bottom_center_y_norm: float


def intersection_area(first: Detection, second: Detection) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def iou(first: Detection, second: Detection) -> float:
    intersection = intersection_area(first, second)
    union = first.bbox_area + second.bbox_area - intersection
    return intersection / union if union > 0 else 0.0


def suppress_nested_detections(
    detections: list[Detection],
    duplicate_iou: float = 0.75,
    containment: float = 0.90,
    max_inner_area_ratio: float = 0.25,
) -> tuple[list[Detection], dict[int, str]]:
    """Remove duplicate and small vehicle boxes contained inside a larger vehicle box."""
    removal_reasons: dict[int, str] = {}

    # First remove almost-identical boxes, regardless of their original COCO class.
    deduplicated: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(iou(candidate, kept) >= duplicate_iou for kept in deduplicated):
            removal_reasons[id(candidate)] = "duplicate_iou"
            continue
        deduplicated.append(candidate)

    # Then remove small boxes that lie mostly inside a larger vehicle box.
    cleaned: list[Detection] = []
    for candidate in sorted(deduplicated, key=lambda item: item.bbox_area, reverse=True):
        remove = False
        for larger in cleaned:
            if larger.bbox_area <= candidate.bbox_area:
                continue
            contained_fraction = intersection_area(candidate, larger) / max(candidate.bbox_area, 1.0)
            area_ratio = candidate.bbox_area / max(larger.bbox_area, 1.0)
            if contained_fraction >= containment and area_ratio <= max_inner_area_ratio:
                removal_reasons[id(candidate)] = "nested_inside_vehicle"
                remove = True
                break
        if not remove:
            cleaned.append(candidate)

    return sorted(cleaned, key=lambda item: item.track_id), removal_reasons


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        60 + (track_id * 47) % 196,
        60 + (track_id * 89) % 196,
        60 + (track_id * 131) % 196,
    )


def draw_detection(frame, detection: Detection) -> None:
    color = color_for_track(detection.track_id)
    point1 = (round(detection.x1), round(detection.y1))
    point2 = (round(detection.x2), round(detection.y2))
    cv2.rectangle(frame, point1, point2, color, 2)
    label = f"ID {detection.track_id} vehicle {detection.confidence:.2f}"
    label_y = max(22, point1[1] - 7)
    cv2.putText(
        frame,
        label,
        (point1[0], label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    bottom_center = (round(detection.bottom_center_x), round(detection.bottom_center_y))
    cv2.circle(frame, bottom_center, 4, color, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track vehicles, remove nested false boxes, and export tracks.csv."
    )
    parser.add_argument("--source", type=Path, required=True, help="Input video path")
    parser.add_argument("--model", type=Path, default=Path("yolo26s.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracking_06_17"))
    parser.add_argument("--start", type=float, default=6.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=17.0, help="End time in seconds")
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    model_path = args.model.resolve()
    output_dir = args.output.resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, round(args.start * fps))
    end_frame = min(total_frames, round(args.end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    video_path = output_dir / "tracked_clean.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {video_path}")

    model = YOLO(str(model_path))
    raw_rows: list[dict] = []
    clean_rows: list[dict] = []
    clean_detections_by_track: dict[int, list[Detection]] = defaultdict(list)
    removed_count = 0
    processed_frames = 0

    try:
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break

            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=VEHICLE_CLASS_IDS,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                quantize=16 if args.device != "cpu" else False,
                agnostic_nms=True,
                nms=True,
                verbose=False,
            )[0]

            timestamp_s = frame_idx / fps
            clip_frame_idx = frame_idx - start_frame
            frame_detections: list[Detection] = []

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy_values = boxes.xyxy.detach().cpu().tolist()
                confidence_values = boxes.conf.detach().cpu().tolist()
                class_values = boxes.cls.detach().cpu().tolist()
                id_values = boxes.id.detach().cpu().tolist()

                for xyxy, confidence, class_id_value, track_id_value in zip(
                    xyxy_values,
                    confidence_values,
                    class_values,
                    id_values,
                    strict=True,
                ):
                    x1, y1, x2, y2 = map(float, xyxy)
                    class_id = int(class_id_value)
                    track_id = int(track_id_value)
                    width = max(0.0, x2 - x1)
                    height = max(0.0, y2 - y1)
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    bottom_center_x = center_x
                    bottom_center_y = y2
                    detection = Detection(
                        run_id=output_dir.name,
                        video_id=source.stem,
                        source_video=str(source),
                        fps=fps,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        frame_idx=frame_idx,
                        frame_number=frame_idx + 1,
                        clip_frame_idx=clip_frame_idx,
                        timestamp_s=timestamp_s,
                        clip_time_s=clip_frame_idx / fps,
                        track_id=track_id,
                        class_id=class_id,
                        class_name=str(result.names[class_id]),
                        object_group="vehicle",
                        confidence=float(confidence),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        center_x=center_x,
                        center_y=center_y,
                        bottom_center_x=bottom_center_x,
                        bottom_center_y=bottom_center_y,
                        bbox_width=width,
                        bbox_height=height,
                        bbox_area=width * height,
                        x1_norm=x1 / frame_width,
                        y1_norm=y1 / frame_height,
                        x2_norm=x2 / frame_width,
                        y2_norm=y2 / frame_height,
                        bottom_center_x_norm=bottom_center_x / frame_width,
                        bottom_center_y_norm=bottom_center_y / frame_height,
                    )
                    frame_detections.append(detection)

            cleaned, removal_reasons = suppress_nested_detections(frame_detections)
            removed_count += len(removal_reasons)

            for detection in frame_detections:
                row = asdict(detection)
                row["kept_after_filter"] = id(detection) not in removal_reasons
                row["filter_reason"] = removal_reasons.get(id(detection), "")
                raw_rows.append(row)

            for detection in cleaned:
                clean_rows.append(asdict(detection))
                clean_detections_by_track[detection.track_id].append(detection)
                draw_detection(frame, detection)

            cv2.putText(
                frame,
                f"source time {timestamp_s:05.2f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            processed_frames += 1

            if args.show:
                cv2.imshow("Clean vehicle tracks", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if processed_frames % 60 == 0:
                print(f"Processed {processed_frames}/{end_frame - start_frame} frames")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    detection_fields = list(Detection.__dataclass_fields__.keys())
    raw_fields = detection_fields + ["kept_after_filter", "filter_reason"]
    write_csv(output_dir / "tracks_raw.csv", raw_rows, raw_fields)
    write_csv(output_dir / "tracks.csv", clean_rows, detection_fields)

    summary_rows = []
    for track_id, detections in sorted(
        clean_detections_by_track.items(), key=lambda item: len(item[1]), reverse=True
    ):
        ordered = sorted(detections, key=lambda detection: detection.frame_idx)
        longest_streak = 0
        current_streak = 0
        previous_frame = None
        for detection in ordered:
            if previous_frame is not None and detection.frame_idx == previous_frame + 1:
                current_streak += 1
            else:
                current_streak = 1
            longest_streak = max(longest_streak, current_streak)
            previous_frame = detection.frame_idx
        summary_rows.append(
            {
                "track_id": track_id,
                "frames_seen": len(ordered),
                "first_timestamp_s": ordered[0].timestamp_s,
                "last_timestamp_s": ordered[-1].timestamp_s,
                "duration_s": ordered[-1].timestamp_s - ordered[0].timestamp_s,
                "mean_confidence": sum(item.confidence for item in ordered) / len(ordered),
                "max_consecutive_frames": longest_streak,
            }
        )

    summary_fields = [
        "track_id",
        "frames_seen",
        "first_timestamp_s",
        "last_timestamp_s",
        "duration_s",
        "mean_confidence",
        "max_consecutive_frames",
    ]
    write_csv(output_dir / "track_summary.csv", summary_rows, summary_fields)

    print()
    print(f"Done: {processed_frames} frames")
    print(f"Raw detections: {len(raw_rows)}")
    print(f"Removed duplicate/nested boxes: {removed_count}")
    print(f"Clean detections: {len(clean_rows)}")
    print(f"Clean tracks: {len(clean_detections_by_track)}")
    print(f"CSV: {output_dir / 'tracks.csv'}")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
