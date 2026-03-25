# run_rep_ablations.py
"""
Representation ablations (4.1 / 4.2 / 4.3 support) with separate outputs.

- Reads your dataset using build_items_multi() (ear, labels_json, color)
- Uses dataset_split.json indices (same as train_yolo_updated.py)
- Exports a YOLO dataset per representation config
- Trains a YOLO model (default yolov8n.pt) per config
- Evaluates on validation using your evaluate_metrics() and saves results

Outputs per config:
  ablation_runs/<cfg_name>/
    dataset/...
    training/...
    metrics_val.json
    metrics_val.csv
"""

import os
import json
import csv
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from yolo_utils import (
    build_items_multi,
    collapse_class_id,
    evaluate_metrics,
    find_best_operating_point,
    YOLO_CLASS_NAMES,
)

# -----------------------------
# Config
# -----------------------------
W, H = 480, 640
SPLIT_JSON = "dataset_split.json"

DEFAULT_MODEL_NAME = "yolov8"
DEFAULT_PRETRAIN = "yolov8n.pt"

BASE_OUT = Path("ablation_runs")


# =========================================================
# Representation generator
# =========================================================

def ear_to_channels(ear: np.ndarray, elev_mode: str) -> np.ndarray:
    """
    Convert EAR cube -> (H,W) float map before normalization.
    elev_mode: 'mean'|'max'|'sum'|'std'
    """
    if ear.ndim == 3:
        if elev_mode == "mean":
            m = ear.mean(axis=0)
        elif elev_mode == "max":
            m = ear.max(axis=0)
        elif elev_mode == "sum":
            m = ear.sum(axis=0)
        elif elev_mode == "std":
            m = ear.std(axis=0)
        else:
            raise ValueError(f"Unknown elev_mode: {elev_mode}")
    else:
        # already 2D
        m = ear
    return m.astype(np.float32)


def normalize_to_uint8(img2d: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Mirrors yolo_utils.load_ear_as_image() pipeline but for arbitrary 2D input:
      log -> resize -> zscore -> minmax -> uint8
    """
    img2d = np.log(img2d + 1e-6)
    img2d = cv2.resize(img2d, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    m = float(img2d.mean())
    s = float(img2d.std()) + 1e-6
    img2d = (img2d - m) / s

    img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-6)
    u8 = (img2d * 255.0).clip(0, 255).astype(np.uint8)
    return u8


def make_radar_image(
    ear_path: str,
    out_w: int,
    out_h: int,
    elev_pool: str = "mean",
    channel_mode: str = "replicate",
) -> np.ndarray:
    """
    Returns uint8 image in shape (H, W, 3).

    elev_pool: 'mean'|'max'|'sum'
      - used for channel_mode='replicate'
    channel_mode:
      - 'replicate': single map (from elev_pool) replicated to 3 channels (your baseline)
      - 'elev_stats': 3 channels from EAR stats: mean/max/std (4.3 practical alternative)
    """
    ear = np.load(ear_path).astype(np.float32)

    if channel_mode == "replicate":
        base = ear_to_channels(ear, elev_pool)
        ch = normalize_to_uint8(base, out_w, out_h)
        img = np.stack([ch, ch, ch], axis=-1)
        return img

    if channel_mode == "elev_stats":
        # 3 distinct channels → not arbitrary: captures different elevation behaviors
        ch1 = normalize_to_uint8(ear_to_channels(ear, "mean"), out_w, out_h)
        ch2 = normalize_to_uint8(ear_to_channels(ear, "max"),  out_w, out_h)
        ch3 = normalize_to_uint8(ear_to_channels(ear, "std"),  out_w, out_h)
        img = np.stack([ch1, ch2, ch3], axis=-1)
        return img

    raise ValueError(f"Unknown channel_mode: {channel_mode}")


# =========================================================
# Exporter (custom, based on your yolo_utils._write_yolo_split)
# =========================================================

def write_yolo_split_custom(split: str, items, out_root: Path, out_w: int, out_h: int,
                            elev_pool: str, channel_mode: str):
    img_dir = out_root / f"images/{split}"
    lab_dir = out_root / f"labels/{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    img_paths = []

    for i, (ear_path, lab_path, _) in enumerate(items):
        img = make_radar_image(ear_path, out_w, out_h, elev_pool=elev_pool, channel_mode=channel_mode)
        name = f"{split}_{i:06d}.png"
        img_path = img_dir / name
        cv2.imwrite(str(img_path), img)

        with open(lab_path, "r") as f:
            data = json.load(f)

        lines = []
        for b in data.get("boxes", []):
            cid = collapse_class_id(int(b["class_id"]))
            x1, y1, x2, y2 = map(float, b["bbox_xyxy"])

            # clip to bounds (your existing logic)
            x1 = max(0.0, min(x1, out_w - 1))
            x2 = max(0.0, min(x2, out_w - 1))
            y1 = max(0.0, min(y1, out_h - 1))
            y2 = max(0.0, min(y2, out_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0

            # normalized YOLO
            lines.append(f"{cid} {xc/out_w:.6f} {yc/out_h:.6f} {w/out_w:.6f} {h/out_h:.6f}")

        with open(lab_dir / name.replace(".png", ".txt"), "w") as f:
            f.write("\n".join(lines))

        img_paths.append(str(img_path))

    return img_paths


def export_dataset_custom(train_items, val_items, out_root: Path, out_w: int, out_h: int,
                          elev_pool: str, channel_mode: str):
    out_root.mkdir(parents=True, exist_ok=True)

    val_paths = write_yolo_split_custom("val", val_items, out_root, out_w, out_h, elev_pool, channel_mode)
    _ = write_yolo_split_custom("train", train_items, out_root, out_w, out_h, elev_pool, channel_mode)

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, nm in YOLO_CLASS_NAMES.items():
            f.write(f"  {i}: {nm}\n")

    return str(yaml_path), val_paths


# =========================================================
# Run configs
# =========================================================

@dataclass
class RepConfig:
    name: str
    elev_pool: str = "mean"           # mean|max|sum
    channel_mode: str = "replicate"   # replicate|elev_stats


ABLATIONS = [
    # 4.1 Elevation aggregation ablation (baseline replicate)
    RepConfig(name="elev_mean", elev_pool="mean", channel_mode="replicate"),
    RepConfig(name="elev_max",  elev_pool="max",  channel_mode="replicate"),
    RepConfig(name="elev_sum",  elev_pool="sum",  channel_mode="replicate"),

    # 4.3 Practical channelization ablation
    RepConfig(name="channels_elev_stats", elev_pool="mean", channel_mode="elev_stats"),
]


def save_metrics(out_dir: Path, metrics: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "metrics_val.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # CSV (flat)
    flat_keys = sorted(metrics.keys())
    with open(out_dir / "metrics_val.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k in flat_keys:
            w.writerow([k, metrics[k]])


def main(root_dir: str):
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    # Load items
    items = build_items_multi(root_dir)
    if not items:
        raise RuntimeError("No (ear,label,color) triples found. Check root_dir.")

    # Load split indices (same logic as train_yolo_updated.py)
    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)
    train_items = [items[i] for i in split["train"]]
    val_items   = [items[i] for i in split["val"]]

    print(f"Using split: {len(train_items)} train / {len(val_items)} val")

    # Run each ablation
    for cfg in ABLATIONS:
        print("\n" + "="*60)
        print(f"ABLATION: {cfg.name} | elev_pool={cfg.elev_pool} | channel_mode={cfg.channel_mode}")
        print("="*60)

        run_dir = BASE_OUT / cfg.name
        ds_dir  = run_dir / "dataset"
        train_dir = run_dir / "training"

        # Export dataset
        yaml_path, val_paths = export_dataset_custom(
            train_items, val_items, ds_dir, W, H,
            elev_pool=cfg.elev_pool, channel_mode=cfg.channel_mode
        )

        # Train (one model by default)
        model = YOLO(DEFAULT_PRETRAIN)

        results = model.train(
            data=yaml_path,
            epochs=180,
            imgsz=(H, W),
            batch=16,
            project=str(train_dir),
            name=DEFAULT_MODEL_NAME,
            exist_ok=True,
        )

        # Select operating point on val (same as your pipeline)


        best = find_best_operating_point(model, val_items, val_paths)
        best_conf = float(best["best_conf"])
        #m = evaluate_metrics(model, val_items, val_paths, iou=0.5, conf=best_conf, verbose=True)


        # Evaluate on val
        m = evaluate_metrics(model, val_items, val_paths, iou_thres=0.5, conf=best_conf, verbose=True)

        # add metadata
        m["ablation_name"] = cfg.name
        m["elev_pool"] = cfg.elev_pool
        m["channel_mode"] = cfg.channel_mode
        m["best_conf"] = float(best_conf)


        save_metrics(run_dir, m)
        print(f"Saved metrics to: {run_dir}")

    print("\nAll ablations complete.")


if __name__ == "__main__":
    # Example:
    # python run_rep_ablations.py "C:\Users\...\realsense_data_OD"
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_rep_ablations.py <root_dir>")
    main(sys.argv[1])
