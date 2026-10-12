from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrackPoint:
    frame_idx: int
    timestamp_s: float
    x: float
    y: float


@dataclass
class WindowSample:
    track_id: int
    segment_id: int
    times: np.ndarray
    obs: np.ndarray
    future: np.ndarray
    split: str = "unused"

    @property
    def start_time(self) -> float:
        return float(self.times[0])

    @property
    def obs_end_time(self) -> float:
        return float(self.times[len(self.obs) - 1])

    @property
    def end_time(self) -> float:
        return float(self.times[-1])


class GRUResidualPredictor(nn.Module):
    """Predict a correction to a constant-velocity trajectory."""

    def __init__(self, input_size: int, hidden_size: int, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, pred_len * 2),
        )
        final_layer = self.head[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, features: torch.Tensor, cv_future: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(features)
        residual = self.head(hidden[-1]).reshape(-1, self.pred_len, 2)
        return cv_future + residual


def parse_id_set(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small GRU demo for surrounding-vehicle image-coordinate "
            "prediction and compare it with motion baselines."
        )
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("outputs/tracking_06_17_v2/tracks.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/surrounding_gru"))
    parser.add_argument("--exclude-id", type=int, default=52)
    parser.add_argument("--train-ids", default="1,2,4")
    parser.add_argument("--val-ids", default="70")
    parser.add_argument("--test-ids", default="292,301,306")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--obs-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_tracks(path: Path, exclude_id: int) -> tuple[dict[int, list[TrackPoint]], dict]:
    tracks: dict[int, list[TrackPoint]] = defaultdict(list)
    metadata: dict = {}
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_index, row in enumerate(reader):
            track_id = int(row["track_id"])
            if row_index == 0:
                metadata = {
                    "source_video": row["source_video"],
                    "video_id": row["video_id"],
                    "fps": float(row["fps"]),
                    "frame_width": int(row["frame_width"]),
                    "frame_height": int(row["frame_height"]),
                }
            if track_id == exclude_id:
                continue
            tracks[track_id].append(
                TrackPoint(
                    frame_idx=int(row["frame_idx"]),
                    timestamp_s=float(row["timestamp_s"]),
                    x=float(row["bottom_center_x_norm"]),
                    y=float(row["bottom_center_y_norm"]),
                )
            )
    for points in tracks.values():
        points.sort(key=lambda point: point.frame_idx)
    return dict(tracks), metadata


def split_on_long_gaps(
    points: list[TrackPoint], max_gap: int
) -> list[list[TrackPoint]]:
    if not points:
        return []
    segments: list[list[TrackPoint]] = []
    current = [points[0]]
    for following in points[1:]:
        previous = current[-1]
        frame_delta = following.frame_idx - previous.frame_idx
        if frame_delta <= 0:
            raise ValueError("Track frames must be strictly increasing")
        missing = frame_delta - 1
        if missing > max_gap:
            segments.append(current)
            current = [following]
            continue
        current.append(following)
    segments.append(current)
    return segments


def resample_segment(
    points: list[TrackPoint], sample_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_times = np.asarray([point.timestamp_s for point in points], dtype=np.float64)
    source_xy = np.asarray([[point.x, point.y] for point in points], dtype=np.float64)
    duration = float(source_times[-1] - source_times[0])
    count = math.floor(duration * sample_hz + 1e-7) + 1
    grid = source_times[0] + np.arange(count, dtype=np.float64) / sample_hz
    grid = grid[grid <= source_times[-1] + 1e-8]
    if len(grid) == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return grid, empty, empty
    label_xy = np.column_stack(
        (
            np.interp(grid, source_times, source_xy[:, 0]),
            np.interp(grid, source_times, source_xy[:, 1]),
        )
    )
    # The observation stream is causal: every 10 Hz point uses only the latest
    # detector output available at that instant. Linear interpolation remains
    # appropriate for the offline future label, but must not leak into inputs.
    causal_indices = np.searchsorted(source_times, grid, side="right") - 1
    causal_indices = np.clip(causal_indices, 0, len(source_times) - 1)
    observation_xy = source_xy[causal_indices]
    return grid, observation_xy, label_xy


def build_windows(
    tracks: dict[int, list[TrackPoint]],
    sample_hz: float,
    obs_len: int,
    pred_len: int,
    max_gap: int,
) -> list[WindowSample]:
    total_len = obs_len + pred_len
    samples: list[WindowSample] = []
    for track_id, points in sorted(tracks.items()):
        for segment_id, segment in enumerate(split_on_long_gaps(points, max_gap)):
            times, observation_xy, label_xy = resample_segment(segment, sample_hz)
            if len(observation_xy) < total_len:
                continue
            for start in range(len(observation_xy) - total_len + 1):
                stop = start + total_len
                samples.append(
                    WindowSample(
                        track_id=track_id,
                        segment_id=segment_id,
                        times=times[start:stop].copy(),
                        obs=observation_xy[start : start + obs_len].copy(),
                        future=label_xy[start + obs_len : stop].copy(),
                    )
                )
    return samples


def assign_splits(
    samples: list[WindowSample],
    train_ids: set[int],
    val_ids: set[int],
    test_ids: set[int],
) -> list[WindowSample]:
    val_candidates = [sample for sample in samples if sample.track_id in val_ids]
    if not val_candidates:
        raise ValueError("No validation windows were created")
    val_start = min(sample.start_time for sample in val_candidates)
    selected: list[WindowSample] = []
    for sample in samples:
        if sample.track_id in train_ids and sample.end_time < val_start:
            sample.split = "train"
        elif sample.track_id in val_ids:
            sample.split = "val"
        elif sample.track_id in test_ids:
            sample.split = "test"
        else:
            continue
        selected.append(sample)
    return selected


def constant_velocity(obs_relative: np.ndarray, dt: float, pred_len: int) -> np.ndarray:
    fit_points = min(5, len(obs_relative))
    fit_xy = obs_relative[-fit_points:]
    fit_t = np.arange(-fit_points + 1, 1, dtype=np.float64) * dt
    design = np.column_stack((fit_t, np.ones_like(fit_t)))
    coefficients, *_ = np.linalg.lstsq(design, fit_xy, rcond=None)
    future_t = np.arange(1, pred_len + 1, dtype=np.float64) * dt
    return future_t[:, None] * coefficients[0][None, :] + coefficients[1][None, :]


def sample_arrays(
    samples: list[WindowSample], sample_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cv_predictions: list[np.ndarray] = []
    dt = 1.0 / sample_hz
    for sample in samples:
        anchor = sample.obs[-1]
        obs_relative = sample.obs - anchor
        velocity = np.zeros_like(obs_relative)
        velocity[1:] = np.diff(obs_relative, axis=0) / dt
        if len(velocity) > 1:
            velocity[0] = velocity[1]
        features.append(np.concatenate((obs_relative, velocity), axis=1))
        targets.append(sample.future - anchor)
        cv_predictions.append(constant_velocity(obs_relative, dt, len(sample.future)))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(cv_predictions, dtype=np.float32),
    )


def fit_scaler(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=(0, 1), keepdims=True)
    std = train_features.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    cv_predictions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
        torch.from_numpy(cv_predictions),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator,
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    item_count = 0
    with torch.no_grad():
        for features, targets, cv_predictions in loader:
            features = features.to(device)
            targets = targets.to(device)
            cv_predictions = cv_predictions.to(device)
            predictions = model(features, cv_predictions)
            loss = loss_fn(predictions, targets)
            weighted_loss += float(loss.item()) * len(features)
            item_count += len(features)
    return weighted_loss / max(1, item_count)


def predict(
    model: nn.Module,
    features: np.ndarray,
    cv_predictions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.from_numpy(features).to(device)
        cv_tensor = torch.from_numpy(cv_predictions).to(device)
        return model(feature_tensor, cv_tensor).cpu().numpy()


def pixel_errors(
    predictions: np.ndarray, targets: np.ndarray, width: int, height: int
) -> np.ndarray:
    scale = np.asarray([width, height], dtype=np.float32)
    return np.linalg.norm((predictions - targets) * scale, axis=-1)


def metric_summary(errors: np.ndarray, image_diagonal: float) -> dict:
    return {
        "ade_px": float(errors.mean()),
        "fde_px": float(errors[:, -1].mean()),
        "error_0_1s_px": float(errors[:, 0].mean()),
        "error_0_5s_px": float(errors[:, min(4, errors.shape[1] - 1)].mean()),
        "error_1_0s_px": float(errors[:, -1].mean()),
        "ade_image_diagonal_fraction": float(errors.mean() / image_diagonal),
    }


def macro_metric_summary(
    errors: np.ndarray,
    samples: list[WindowSample],
    image_diagonal: float,
) -> dict:
    per_track: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        per_track[sample.track_id].append(index)
    summaries = [
        metric_summary(errors[indices], image_diagonal) for indices in per_track.values()
    ]
    return {
        key: float(np.mean([summary[key] for summary in summaries]))
        for key in summaries[0]
    }


def write_manifest(path: Path, samples: Iterable[WindowSample]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "track_id",
                "segment_id",
                "start_time_s",
                "obs_end_time_s",
                "future_end_time_s",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "split": sample.split,
                    "track_id": sample.track_id,
                    "segment_id": sample.segment_id,
                    "start_time_s": sample.start_time,
                    "obs_end_time_s": sample.obs_end_time,
                    "future_end_time_s": sample.end_time,
                }
            )


def write_predictions(
    path: Path,
    samples: list[WindowSample],
    targets: np.ndarray,
    cv_predictions: np.ndarray,
    gru_predictions: np.ndarray,
    width: int,
    height: int,
) -> None:
    fields = [
        "track_id",
        "segment_id",
        "obs_end_time_s",
        "horizon_step",
        "future_time_s",
        "anchor_x_norm",
        "anchor_y_norm",
        "actual_x_norm",
        "actual_y_norm",
        "cv_x_norm",
        "cv_y_norm",
        "gru_x_norm",
        "gru_y_norm",
        "cv_error_px",
        "gru_error_px",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for sample_index, sample in enumerate(samples):
            anchor = sample.obs[-1]
            for horizon in range(targets.shape[1]):
                actual = targets[sample_index, horizon] + anchor
                cv_point = cv_predictions[sample_index, horizon] + anchor
                gru_point = gru_predictions[sample_index, horizon] + anchor
                cv_error = math.hypot(
                    (cv_point[0] - actual[0]) * width,
                    (cv_point[1] - actual[1]) * height,
                )
                gru_error = math.hypot(
                    (gru_point[0] - actual[0]) * width,
                    (gru_point[1] - actual[1]) * height,
                )
                writer.writerow(
                    {
                        "track_id": sample.track_id,
                        "segment_id": sample.segment_id,
                        "obs_end_time_s": sample.obs_end_time,
                        "horizon_step": horizon + 1,
                        "future_time_s": sample.times[len(sample.obs) + horizon],
                        "anchor_x_norm": anchor[0],
                        "anchor_y_norm": anchor[1],
                        "actual_x_norm": actual[0],
                        "actual_y_norm": actual[1],
                        "cv_x_norm": cv_point[0],
                        "cv_y_norm": cv_point[1],
                        "gru_x_norm": gru_point[0],
                        "gru_y_norm": gru_point[1],
                        "cv_error_px": cv_error,
                        "gru_error_px": gru_error,
                    }
                )


def write_per_track_metrics(
    path: Path,
    samples: list[WindowSample],
    error_sets: dict[str, np.ndarray],
    image_diagonal: float,
) -> None:
    by_track: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_track[sample.track_id].append(index)
    fields = ["track_id", "windows"]
    for model_name in error_sets:
        fields.extend([f"{model_name}_ade_px", f"{model_name}_fde_px"])
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for track_id, indices in sorted(by_track.items()):
            row: dict[str, float | int] = {
                "track_id": track_id,
                "windows": len(indices),
            }
            for model_name, errors in error_sets.items():
                summary = metric_summary(errors[indices], image_diagonal)
                row[f"{model_name}_ade_px"] = summary["ade_px"]
                row[f"{model_name}_fde_px"] = summary["fde_px"]
            writer.writerow(row)


def plot_losses(path: Path, train_losses: list[float], val_losses: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="train")
    axis.plot(epochs, val_losses, label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Smooth L1 loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_trajectories(
    path: Path,
    samples: list[WindowSample],
    cv_predictions: np.ndarray,
    gru_predictions: np.ndarray,
    width: int,
    height: int,
) -> None:
    column_count = 3
    row_count = math.ceil(len(samples) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, 3.7 * row_count),
        squeeze=False,
    )
    for index, axis in enumerate(axes.flat):
        if index >= len(samples):
            axis.axis("off")
            continue
        sample = samples[index]
        anchor = sample.obs[-1]
        scale = np.asarray([width, height], dtype=np.float64)
        observed = sample.obs * scale
        actual = sample.future * scale
        cv_absolute = (cv_predictions[index] + anchor) * scale
        gru_absolute = (gru_predictions[index] + anchor) * scale
        axis.plot(observed[:, 0], observed[:, 1], "o-k", label="observed", markersize=3)
        axis.plot(actual[:, 0], actual[:, 1], "o-g", label="actual", markersize=3)
        axis.plot(cv_absolute[:, 0], cv_absolute[:, 1], "--b", label="CV")
        axis.plot(gru_absolute[:, 0], gru_absolute[:, 1], "--r", label="GRU")
        axis.set_title(f"track {sample.track_id} | obs end {sample.obs_end_time:.2f}s")
        combined = np.vstack((observed, actual, cv_absolute, gru_absolute))
        center_x = float((combined[:, 0].min() + combined[:, 0].max()) / 2)
        center_y = float((combined[:, 1].min() + combined[:, 1].max()) / 2)
        span_x = max(70.0, float(np.ptp(combined[:, 0])) * 1.35)
        span_y = max(50.0, float(np.ptp(combined[:, 1])) * 1.35)
        axis.set_xlim(center_x - span_x / 2, center_x + span_x / 2)
        axis.set_ylim(center_y + span_y / 2, center_y - span_y / 2)
        axis.grid(alpha=0.2)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (pixels)")
        axis.set_ylabel("y (pixels)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.suptitle("Image-coordinate trajectory prediction (test windows)", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tracks_path = args.tracks.resolve()
    output_dir = args.output.resolve()
    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = parse_id_set(args.train_ids)
    val_ids = parse_id_set(args.val_ids)
    test_ids = parse_id_set(args.test_ids)
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Train, validation, and test track IDs must be disjoint")

    tracks, metadata = read_tracks(tracks_path, args.exclude_id)
    windows = build_windows(
        tracks,
        args.sample_hz,
        args.obs_len,
        args.pred_len,
        args.max_gap,
    )
    selected = assign_splits(windows, train_ids, val_ids, test_ids)
    split_samples = {
        split: [sample for sample in selected if sample.split == split]
        for split in ("train", "val", "test")
    }
    if any(not samples for samples in split_samples.values()):
        raise ValueError(
            "Every split needs at least one window: "
            + str({key: len(value) for key, value in split_samples.items()})
        )

    arrays = {
        split: sample_arrays(samples, args.sample_hz)
        for split, samples in split_samples.items()
    }
    scaler_mean, scaler_std = fit_scaler(arrays["train"][0])
    scaled_features = {
        split: ((values[0] - scaler_mean) / scaler_std).astype(np.float32)
        for split, values in arrays.items()
    }

    loaders = {
        split: make_loader(
            scaled_features[split],
            arrays[split][1],
            arrays[split][2],
            args.batch_size,
            split == "train",
            args.seed,
        )
        for split in ("train", "val", "test")
    }

    device = torch.device(args.device)
    model = GRUResidualPredictor(4, args.hidden_size, args.pred_len).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=0.005)

    best_epoch = 0
    best_val_loss = evaluate_loss(model, loaders["val"], loss_fn, device)
    best_state = deepcopy(model.state_dict())
    train_losses: list[float] = []
    val_losses: list[float] = []
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        item_count = 0
        for features, targets, cv_predictions in loaders["train"]:
            features = features.to(device)
            targets = targets.to(device)
            cv_predictions = cv_predictions.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features, cv_predictions)
            loss = loss_fn(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(features)
            item_count += len(features)
        train_loss = total_loss / item_count
        val_loss = evaluate_loss(model, loaders["val"], loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-8:
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    test_samples = split_samples["test"]
    test_targets = arrays["test"][1]
    test_cv = arrays["test"][2]
    test_gru = predict(model, scaled_features["test"], test_cv, device)
    test_last = np.zeros_like(test_targets)

    width = metadata["frame_width"]
    height = metadata["frame_height"]
    image_diagonal = math.hypot(width, height)
    error_sets = {
        "last_position": pixel_errors(test_last, test_targets, width, height),
        "constant_velocity": pixel_errors(test_cv, test_targets, width, height),
        "gru": pixel_errors(test_gru, test_targets, width, height),
    }
    metrics = {
        "warning": (
            "Demo only: one moving-camera scene and seven usable surrounding tracks. "
            "Errors are image pixels, not world-space meters."
        ),
        "data": {
            **metadata,
            "tracks_csv": str(tracks_path),
            "excluded_track_id": args.exclude_id,
            "usable_track_ids": sorted({sample.track_id for sample in selected}),
            "all_eligible_windows": len(windows),
            "selected_windows": len(selected),
            "split_windows": {
                split: len(samples) for split, samples in split_samples.items()
            },
            "split_track_ids": {
                split: sorted({sample.track_id for sample in samples})
                for split, samples in split_samples.items()
            },
            "split_time_ranges_s": {
                split: [
                    min(sample.start_time for sample in samples),
                    max(sample.end_time for sample in samples),
                ]
                for split, samples in split_samples.items()
            },
        },
        "preprocessing": {
            "coordinate": "bottom_center normalized image coordinate",
            "sample_hz": args.sample_hz,
            "observation_steps": args.obs_len,
            "prediction_steps": args.pred_len,
            "max_missing_frames_within_segment": args.max_gap,
            "observation_resampling": "causal latest-past detection",
            "future_label_resampling": "offline linear interpolation",
            "feature_names": ["relative_x", "relative_y", "velocity_x", "velocity_y"],
        },
        "training": {
            "seed": args.seed,
            "device": str(device),
            "hidden_size": args.hidden_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "epochs_run": len(train_losses),
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
        },
        "test_micro": {
            name: metric_summary(errors, image_diagonal)
            for name, errors in error_sets.items()
        },
        "test_track_macro": {
            name: macro_metric_summary(errors, test_samples, image_diagonal)
            for name, errors in error_sets.items()
        },
    }

    config = {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_manifest(output_dir / "sample_manifest.csv", selected)
    write_predictions(
        output_dir / "predictions.csv",
        test_samples,
        test_targets,
        test_cv,
        test_gru,
        width,
        height,
    )
    write_per_track_metrics(
        output_dir / "metrics_per_track.csv",
        test_samples,
        error_sets,
        image_diagonal,
    )
    plot_losses(output_dir / "loss_curve.png", train_losses, val_losses)
    plot_trajectories(
        output_dir / "trajectory_predictions.png",
        test_samples,
        test_cv,
        test_gru,
        width,
        height,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": scaler_mean,
            "feature_std": scaler_std,
            "config": config,
            "metadata": metadata,
        },
        output_dir / "model.pt",
    )

    print(f"Device: {device}")
    print(
        "Windows: "
        + ", ".join(
            f"{split}={len(samples)}" for split, samples in split_samples.items()
        )
    )
    print(f"Best epoch: {best_epoch} / validation loss: {best_val_loss:.8f}")
    for name, summary in metrics["test_micro"].items():
        print(
            f"{name:>17}: ADE={summary['ade_px']:.2f}px, "
            f"FDE={summary['fde_px']:.2f}px"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
