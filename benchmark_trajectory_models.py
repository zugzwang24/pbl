from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from train_surrounding_gru import (
    GRUResidualPredictor,
    WindowSample,
    assign_splits,
    build_windows,
    evaluate_loss,
    fit_scaler,
    macro_metric_summary,
    make_loader,
    metric_summary,
    pixel_errors,
    predict,
    parse_id_set,
    read_tracks,
    sample_arrays,
    set_seed,
)


class LinearResidualPredictor(nn.Module):
    def __init__(self, obs_len: int, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output = nn.Linear(obs_len * 4, pred_len * 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, cv_future: torch.Tensor) -> torch.Tensor:
        residual = self.output(features.flatten(start_dim=1)).reshape(
            -1, self.pred_len, 2
        )
        return cv_future + residual


class MLPResidualPredictor(nn.Module):
    def __init__(self, obs_len: int, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.network = nn.Sequential(
            nn.Linear(obs_len * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, pred_len * 2),
        )
        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, features: torch.Tensor, cv_future: torch.Tensor) -> torch.Tensor:
        residual = self.network(features.flatten(start_dim=1)).reshape(
            -1, self.pred_len, 2
        )
        return cv_future + residual


class LSTMResidualPredictor(nn.Module):
    def __init__(self, hidden_size: int, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)
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
        _, (hidden, _) = self.lstm(features)
        residual = self.head(hidden[-1]).reshape(-1, self.pred_len, 2)
        return cv_future + residual


class TransformerResidualPredictor(nn.Module):
    def __init__(self, obs_len: int, pred_len: int, d_model: int = 16) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.input_projection = nn.Linear(4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=64,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.register_buffer(
            "positional_encoding",
            self._sinusoidal_encoding(obs_len, d_model),
            persistent=True,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, pred_len * 2),
        )
        final_layer = self.head[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    @staticmethod
    def _sinusoidal_encoding(length: int, width: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / width)
        )
        encoding = torch.zeros(1, length, width, dtype=torch.float32)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        encoding[0, :, 1::2] = torch.cos(positions * frequencies)
        return encoding

    def forward(self, features: torch.Tensor, cv_future: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(features) + self.positional_encoding
        encoded = self.encoder(encoded)
        residual = self.head(encoded[:, -1]).reshape(-1, self.pred_len, 2)
        return cv_future + residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare physics and neural trajectory predictors on one fixed split."
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("outputs/tracking_custom_v1/tracks.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/model_benchmark_correct_v1")
    )
    parser.add_argument("--exclude-id", type=int, default=5)
    parser.add_argument("--train-ids", default="1,2,4,6,19")
    parser.add_argument("--val-ids", default="363")
    parser.add_argument("--test-ids", default="369")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--obs-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def constant_acceleration_predictions(
    samples: list[WindowSample], sample_hz: float, pred_len: int
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    dt = 1.0 / sample_hz
    for sample in samples:
        anchor = sample.obs[-1]
        obs_relative = sample.obs - anchor
        fit_points = min(7, len(obs_relative))
        fit_xy = obs_relative[-fit_points:]
        fit_t = np.arange(-fit_points + 1, 1, dtype=np.float64) * dt
        design = np.column_stack((0.5 * fit_t**2, fit_t, np.ones_like(fit_t)))
        coefficients, *_ = np.linalg.lstsq(design, fit_xy, rcond=None)
        future_t = np.arange(1, pred_len + 1, dtype=np.float64) * dt
        future_design = np.column_stack(
            (0.5 * future_t**2, future_t, np.ones_like(future_t))
        )
        predictions.append(future_design @ coefficients)
    return np.asarray(predictions, dtype=np.float32)


def kalman_cv_prediction(
    sample: WindowSample,
    sample_hz: float,
    pred_len: int,
    acceleration_variance: float,
    measurement_variance: float,
) -> np.ndarray:
    dt = 1.0 / sample_hz
    anchor = sample.obs[-1]
    observations = sample.obs - anchor
    transition = np.asarray(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observation_matrix = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
    )
    axis_process = np.asarray(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
        dtype=np.float64,
    ) * acceleration_variance
    process_noise = np.asarray(
        [
            [axis_process[0, 0], 0.0, axis_process[0, 1], 0.0],
            [0.0, axis_process[0, 0], 0.0, axis_process[0, 1]],
            [axis_process[1, 0], 0.0, axis_process[1, 1], 0.0],
            [0.0, axis_process[1, 0], 0.0, axis_process[1, 1]],
        ],
        dtype=np.float64,
    )
    measurement_noise = np.eye(2, dtype=np.float64) * measurement_variance
    initial_velocity = (observations[1] - observations[0]) / dt
    state = np.asarray(
        [observations[0, 0], observations[0, 1], initial_velocity[0], initial_velocity[1]],
        dtype=np.float64,
    )
    covariance = np.diag(
        [measurement_variance, measurement_variance, 1e-2, 1e-2]
    ).astype(np.float64)
    identity = np.eye(4, dtype=np.float64)

    for index, measurement in enumerate(observations):
        if index > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        innovation = measurement - observation_matrix @ state
        innovation_covariance = (
            observation_matrix @ covariance @ observation_matrix.T + measurement_noise
        )
        gain = (
            covariance
            @ observation_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        state = state + gain @ innovation
        covariance = (identity - gain @ observation_matrix) @ covariance

    future: list[np.ndarray] = []
    for _ in range(pred_len):
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise
        future.append(state[:2].copy())
    return np.asarray(future, dtype=np.float32)


def kalman_batch(
    samples: list[WindowSample],
    sample_hz: float,
    pred_len: int,
    acceleration_variance: float,
    measurement_variance: float,
) -> np.ndarray:
    return np.asarray(
        [
            kalman_cv_prediction(
                sample,
                sample_hz,
                pred_len,
                acceleration_variance,
                measurement_variance,
            )
            for sample in samples
        ],
        dtype=np.float32,
    )


def tune_kalman(
    samples: list[WindowSample],
    targets: np.ndarray,
    sample_hz: float,
    pred_len: int,
) -> tuple[float, float, float]:
    q_values = (1e-6, 1e-5, 1e-4, 1e-3)
    r_values = (1e-7, 1e-6, 1e-5, 1e-4)
    best: tuple[float, float, float] | None = None
    for q_value in q_values:
        for r_value in r_values:
            predictions = kalman_batch(
                samples, sample_hz, pred_len, q_value, r_value
            )
            absolute_error = np.abs(predictions - targets)
            beta = 0.005
            smooth_l1 = np.where(
                absolute_error < beta,
                0.5 * absolute_error**2 / beta,
                absolute_error - 0.5 * beta,
            )
            validation_loss = float(smooth_l1.mean())
            candidate = (validation_loss, q_value, r_value)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2], best[0]


def train_once(
    model: nn.Module,
    seed: int,
    features: dict[str, np.ndarray],
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, int, float, int, float]:
    loaders = {
        split: make_loader(
            features[split],
            arrays[split][1],
            arrays[split][2],
            args.batch_size,
            split == "train",
            seed,
        )
        for split in ("train", "val", "test")
    }
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=0.005)
    best_epoch = 0
    best_val_loss = evaluate_loss(model, loaders["val"], loss_fn, device)
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    started_at = time.perf_counter()
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        model.train()
        for batch_features, targets, cv_predictions in loaders["train"]:
            batch_features = batch_features.to(device)
            targets = targets.to(device)
            cv_predictions = cv_predictions.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features, cv_predictions)
            loss = loss_fn(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        val_loss = evaluate_loss(model, loaders["val"], loss_fn, device)
        if val_loss < best_val_loss - 1e-8:
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    elapsed = time.perf_counter() - started_at
    model.load_state_dict(best_state)
    return model, best_epoch, best_val_loss, epochs_run, elapsed


def evaluate_prediction(
    model_name: str,
    kind: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    samples: list[WindowSample],
    width: int,
    height: int,
    seed: int | None = None,
    parameters: int = 0,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
    epochs_run: int | None = None,
    train_seconds: float | None = None,
) -> dict:
    diagonal = math.hypot(width, height)
    errors = pixel_errors(predictions, targets, width, height)
    micro = metric_summary(errors, diagonal)
    macro = macro_metric_summary(errors, samples, diagonal)
    return {
        "model": model_name,
        "kind": kind,
        "seed": seed,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_run": epochs_run,
        "train_seconds": train_seconds,
        "micro_ade_px": micro["ade_px"],
        "micro_fde_px": micro["fde_px"],
        "macro_ade_px": macro["ade_px"],
        "macro_fde_px": macro["fde_px"],
        "error_0_1s_px": micro["error_0_1s_px"],
        "error_0_5s_px": micro["error_0_5s_px"],
        "error_1_0s_px": micro["error_1_0s_px"],
    }


def summarize_runs(runs: list[dict], model_order: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["model"]].append(run)
    summaries: list[dict] = []
    metric_names = ("micro_ade_px", "micro_fde_px", "macro_ade_px", "macro_fde_px")
    for model_name in model_order:
        model_runs = grouped[model_name]
        summary: dict[str, float | int | str] = {
            "model": model_name,
            "kind": model_runs[0]["kind"],
            "runs": len(model_runs),
            "parameters": model_runs[0]["parameters"],
        }
        for metric_name in metric_names:
            values = np.asarray([run[metric_name] for run in model_runs], dtype=np.float64)
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    path: Path,
    summaries: list[dict],
    test_track_count: int,
    test_window_count: int,
) -> None:
    labels = [summary["model"] for summary in summaries]
    x_positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    for axis, metric, title in (
        (axes[0], "macro_ade_px", "Track-macro ADE"),
        (axes[1], "macro_fde_px", "Track-macro FDE"),
    ):
        means = [summary[f"{metric}_mean"] for summary in summaries]
        stds = [summary[f"{metric}_std"] for summary in summaries]
        colors = ["#7f8c8d" if summary["kind"] == "physics" else "#3566b8" for summary in summaries]
        axis.bar(x_positions, means, yerr=stds, capsize=4, color=colors, alpha=0.9)
        axis.set_title(title)
        axis.set_ylabel("Pixel error (lower is better)")
        axis.set_xticks(x_positions, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Trajectory model comparison — "
        f"{test_track_count} test track(s) / {test_window_count} overlapping windows\n"
        "Neural error bars show variation across training seeds, not data uncertainty"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tracks_path = args.tracks.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one training seed is required")
    device = torch.device(args.device)

    train_ids = parse_id_set(args.train_ids)
    val_ids = parse_id_set(args.val_ids)
    test_ids = parse_id_set(args.test_ids)
    if not train_ids or not val_ids or not test_ids:
        raise ValueError("Train, validation, and test ID sets must all be non-empty")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Train, validation, and test IDs must be disjoint")
    if args.exclude_id in train_ids | val_ids | test_ids:
        raise ValueError("The excluded track ID cannot appear in a data split")
    if args.sample_hz <= 0 or args.obs_len <= 0 or args.pred_len <= 0:
        raise ValueError("sample-hz, obs-len, and pred-len must be positive")
    if args.max_gap < 0:
        raise ValueError("max-gap must be non-negative")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    sample_hz = args.sample_hz
    obs_len = args.obs_len
    pred_len = args.pred_len
    tracks, metadata = read_tracks(tracks_path, exclude_id=args.exclude_id)
    windows = build_windows(
        tracks, sample_hz, obs_len, pred_len, max_gap=args.max_gap
    )
    selected = assign_splits(
        windows,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
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
        split: sample_arrays(samples, sample_hz)
        for split, samples in split_samples.items()
    }
    scaler_mean, scaler_std = fit_scaler(arrays["train"][0])
    features = {
        split: ((values[0] - scaler_mean) / scaler_std).astype(np.float32)
        for split, values in arrays.items()
    }

    width = metadata["frame_width"]
    height = metadata["frame_height"]
    test_samples = split_samples["test"]
    test_targets = arrays["test"][1]
    test_cv = arrays["test"][2]
    runs: list[dict] = []

    baseline_predictions = {
        "Last position": np.zeros_like(test_targets),
        "CV linear fit": test_cv,
        "CA quadratic fit": constant_acceleration_predictions(
            test_samples, sample_hz, pred_len
        ),
    }
    best_q, best_r, kalman_val_loss = tune_kalman(
        split_samples["val"],
        arrays["val"][1],
        sample_hz,
        pred_len,
    )
    baseline_predictions["Kalman CV"] = kalman_batch(
        test_samples, sample_hz, pred_len, best_q, best_r
    )
    for model_name, predictions in baseline_predictions.items():
        result = evaluate_prediction(
            model_name,
            "physics",
            predictions,
            test_targets,
            test_samples,
            width,
            height,
        )
        runs.append(result)
        print(
            f"{model_name:>18}: macro ADE={result['macro_ade_px']:.2f}px, "
            f"FDE={result['macro_fde_px']:.2f}px",
            flush=True,
        )

    model_factories: dict[str, Callable[[], nn.Module]] = {
        "Linear residual": lambda: LinearResidualPredictor(obs_len, pred_len),
        "MLP residual": lambda: MLPResidualPredictor(obs_len, pred_len),
        "GRU residual": lambda: GRUResidualPredictor(4, 32, pred_len),
        "LSTM residual": lambda: LSTMResidualPredictor(28, pred_len),
        "Transformer residual": lambda: TransformerResidualPredictor(obs_len, pred_len),
    }
    for model_name, factory in model_factories.items():
        for seed in seeds:
            set_seed(seed)
            model = factory()
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            model, best_epoch, best_val_loss, epochs_run, elapsed = train_once(
                model, seed, features, arrays, args, device
            )
            predictions = predict(model, features["test"], test_cv, device)
            result = evaluate_prediction(
                model_name,
                "learned",
                predictions,
                test_targets,
                test_samples,
                width,
                height,
                seed=seed,
                parameters=parameter_count,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                epochs_run=epochs_run,
                train_seconds=elapsed,
            )
            runs.append(result)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": torch.from_numpy(scaler_mean),
                    "feature_std": torch.from_numpy(scaler_std),
                    "model_name": model_name,
                    "seed": seed,
                    "obs_len": obs_len,
                    "pred_len": pred_len,
                },
                checkpoint_dir / f"{model_name.lower().replace(' ', '_')}_seed{seed}.pt",
            )
            print(
                f"{model_name:>18} seed={seed:>3}: "
                f"macro ADE={result['macro_ade_px']:.2f}px, "
                f"FDE={result['macro_fde_px']:.2f}px, "
                f"best epoch={best_epoch}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    model_order = [*baseline_predictions.keys(), *model_factories.keys()]
    summaries = summarize_runs(runs, model_order)
    summaries.sort(key=lambda row: float(row["macro_ade_px_mean"]))
    test_track_count = len({sample.track_id for sample in test_samples})
    test_window_count = len(test_samples)
    write_csv(output_dir / "benchmark_runs.csv", runs)
    write_csv(output_dir / "benchmark_summary.csv", summaries)
    plot_comparison(
        output_dir / "model_comparison.png",
        summaries,
        test_track_count,
        test_window_count,
    )

    payload = {
        "warning": (
            "Exploratory comparison only: one moving-camera scene, "
            f"{test_track_count} test track(s), and {test_window_count} heavily "
            "overlapping test windows. Pixel errors are not meters."
        ),
        "data": {
            **metadata,
            "tracks_csv": str(tracks_path),
            "excluded_track_id": args.exclude_id,
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
            "causal_observations": True,
        },
        "training": {
            "seeds": seeds,
            "device": str(device),
            "sample_hz": sample_hz,
            "observation_steps": obs_len,
            "prediction_steps": pred_len,
            "max_gap_frames": args.max_gap,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "SmoothL1 beta=0.005",
            "shared_prior": "CV linear-fit + learned residual",
        },
        "kalman_validation_tuning": {
            "acceleration_variance": best_q,
            "measurement_variance": best_r,
            "validation_smooth_l1": kalman_val_loss,
        },
        "summary_sorted_by_macro_ade": summaries,
        "runs": runs,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTrack-macro ADE ranking", flush=True)
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"{rank:>2}. {summary['model']:<22} "
            f"ADE={summary['macro_ade_px_mean']:.2f}"
            f"±{summary['macro_ade_px_std']:.2f}px, "
            f"FDE={summary['macro_fde_px_mean']:.2f}"
            f"±{summary['macro_fde_px_std']:.2f}px",
            flush=True,
        )
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
