#!/usr/bin/env python3
"""
Prepare MIT-BIH beat classification dataset for Gilgamesh.

Outputs four files in --out-dir:
  - train_images.npy  [N, 36] float32
  - train_labels.npy  [N] int64
  - test_images.npy   [M, 36] float32
  - test_labels.npy   [M] int64

Class mapping (AAMI 5-class):
  0: N (normal/bundle/escape)
  1: S (supraventricular ectopic)
  2: V (ventricular ectopic)
  3: F (fusion)
  4: Q (paced/unknown)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import wfdb


DS1 = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]

DS2 = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]


def aami_label(symbol: str) -> int | None:
    if symbol in {"N", "L", "R", "e", "j"}:
        return 0
    if symbol in {"A", "a", "J", "S"}:
        return 1
    if symbol in {"V", "E"}:
        return 2
    if symbol in {"F"}:
        return 3
    # Merge rare Q/unknown/paced beats into class 0 to avoid an effectively
    # unlearnable tail class under the 36-9-10 hardware-constrained budget.
    if symbol in {"/", "f", "Q"}:
        return 0
    return None


def _normalize_segment(segment: np.ndarray) -> np.ndarray:
    x = segment.astype(np.float32)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std > 1e-6:
        x = (x - mean) / std
    else:
        x = x - mean
    return x


def downsample_to_36(segment: np.ndarray, feature_mode: str) -> np.ndarray:
    # segment expected length 360
    x = _normalize_segment(segment)

    if feature_mode == "mean_dx":
        # 18 low-rate morphology features + 18 slope features.
        raw_ds = x.reshape(18, 20).mean(axis=1)
        dx = np.diff(x, prepend=x[0])
        dx_ds = dx.reshape(18, 20).mean(axis=1)
        feat = np.concatenate([raw_ds, dx_ds], axis=0)
        return np.clip(feat, -5.0, 5.0)

    if feature_mode == "resample36":
        # Uniform 36-point resampling keeps more waveform morphology than
        # block means while still fitting the 36-input hardware budget.
        src = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
        dst = np.linspace(0.0, 1.0, num=36, dtype=np.float32)
        feat = np.interp(dst, src, x).astype(np.float32)
        return np.clip(feat, -5.0, 5.0)

    if feature_mode == "resample_dx":
        # 18 uniformly resampled morphology + 18 uniformly resampled slope.
        src = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
        dst = np.linspace(0.0, 1.0, num=18, dtype=np.float32)
        raw = np.interp(dst, src, x).astype(np.float32)
        dx = np.diff(x, prepend=x[0])
        dx_r = np.interp(dst, src, dx).astype(np.float32)
        feat = np.concatenate([raw, dx_r], axis=0)
        return np.clip(feat, -5.0, 5.0)

    if feature_mode == "multiscale":
        # Multi-scale morphology summary:
        # 12 coarse means + 12 fine means + 12 local slopes.
        coarse = x.reshape(12, 30).mean(axis=1)
        fine = x.reshape(12, 30)[:, 10:20].mean(axis=1)
        dx = np.diff(x, prepend=x[0])
        slopes = dx.reshape(12, 30).mean(axis=1)
        feat = np.concatenate([coarse, fine, slopes], axis=0)
        return np.clip(feat, -5.0, 5.0)

    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def collect_records(
    record_ids: list[str],
    half_window: int,
    max_per_class: int,
    db_dir: str | None,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    counts: dict[int, int] = defaultdict(int)

    for rid in record_ids:
        if db_dir:
            record_path = str(Path(db_dir) / rid)
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, "atr")
        else:
            record = wfdb.rdrecord(rid, pn_dir="mitdb")
            ann = wfdb.rdann(rid, "atr", pn_dir="mitdb")
        sig = record.p_signal[:, 0]  # Upper channel (usually MLII)
        n = len(sig)

        for idx, sym in zip(ann.sample, ann.symbol):
            cls = aami_label(sym)
            if cls is None:
                continue
            if counts[cls] >= max_per_class:
                continue
            start = idx - half_window
            end = idx + half_window
            if start < 0 or end > n:
                continue
            seg = sig[start:end]
            if seg.shape[0] != 2 * half_window:
                continue
            feat = downsample_to_36(seg, feature_mode)
            features.append(feat)
            labels.append(cls)
            counts[cls] += 1

    x = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data_ecg_36")
    parser.add_argument("--half-window", type=int, default=180)
    parser.add_argument("--train-max-per-class", type=int, default=12000)
    parser.add_argument("--test-max-per-class", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        default="mean_dx",
        choices=["mean_dx", "resample36", "resample_dx", "multiscale"],
        help="36-D feature extraction method.",
    )
    parser.add_argument(
        "--db-dir",
        default=None,
        help="Local MIT-BIH directory with record files (e.g. data_ecg_raw/mitdb).",
    )
    parser.add_argument(
        "--train-records",
        default=None,
        help="Comma-separated train record ids to override default DS1 split.",
    )
    parser.add_argument(
        "--test-records",
        default=None,
        help="Comma-separated test record ids to override default DS2 split.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = (
        [r.strip() for r in args.train_records.split(",") if r.strip()]
        if args.train_records
        else DS1
    )
    test_records = (
        [r.strip() for r in args.test_records.split(",") if r.strip()]
        if args.test_records
        else DS2
    )

    print("Collecting DS1 (train)...")
    x_train, y_train = collect_records(
        train_records,
        half_window=args.half_window,
        max_per_class=args.train_max_per_class,
        db_dir=args.db_dir,
        feature_mode=args.feature_mode,
    )
    print("Collecting DS2 (test)...")
    x_test, y_test = collect_records(
        test_records,
        half_window=args.half_window,
        max_per_class=args.test_max_per_class,
        db_dir=args.db_dir,
        feature_mode=args.feature_mode,
    )

    rng = np.random.default_rng(args.seed)
    train_perm = rng.permutation(len(y_train))
    test_perm = rng.permutation(len(y_test))
    x_train, y_train = x_train[train_perm], y_train[train_perm]
    x_test, y_test = x_test[test_perm], y_test[test_perm]

    np.save(out_dir / "train_images.npy", x_train)
    np.save(out_dir / "train_labels.npy", y_train)
    np.save(out_dir / "test_images.npy", x_test)
    np.save(out_dir / "test_labels.npy", y_test)

    print("\nSaved:")
    print(f"  feature_mode={args.feature_mode}")
    print(f"  {out_dir}/train_images.npy {x_train.shape}")
    print(f"  {out_dir}/train_labels.npy {y_train.shape}")
    print(f"  {out_dir}/test_images.npy  {x_test.shape}")
    print(f"  {out_dir}/test_labels.npy  {y_test.shape}")

    for split, y in [("train", y_train), ("test", y_test)]:
        vals, cnts = np.unique(y, return_counts=True)
        dist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts))
        print(f"  {split} class dist -> {dist}")


if __name__ == "__main__":
    main()
