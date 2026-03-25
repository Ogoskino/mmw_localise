# label_noise_proxy.py
"""
Quantify cross-modal supervision noise proxy (Section 3.1 idea):
Distance between a radar-derived energy centroid and the center of GT camera boxes.

Works directly on your dataset structure:
  (ear_path, labels_json/*.json, color_frame)

Uses the SAME EAR->2D processing as yolo_utils.load_ear_as_image(),
but keeps the normalized float map to compute a stable centroid.

Outputs:
  - per_frame_metrics.csv
  - summary.json
  - histogram.png (optional)
"""

import os
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from yolo_utils import build_items_multi, collapse_class_id  # uses your existing mapping


def load_ear_norm_map(ear_path: str, out_w: int, out_h: int) -> np.ndarray:
    """
    Mirror of yolo_utils.load_ear_as_image() but returns float32 map in [0,1]
    (before uint8 quantization). :contentReference[oaicite:4]{index=4}
    """
    ear = np.load(ear_path).astype(np.float32)

    # EAR cube -> 2D map (mean over elevation if 3D)
    if ear.ndim == 3:
        ear2d = ear.mean(axis=0)
    else:
        ear2d = ear

    # log compression
    ear2d = np.log(ear2d + 1e-6)

    # resize to (W,H) in OpenCV order (width,height)
    ear_resized = cv2.resize(ear2d, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # per-frame standardization + min-max to [0,1]
    m = float(ear_resized.mean())
    s = float(ear_resized.std()) + 1e-6
    ear_norm = (ear_resized - m) / s
    ear_norm = (ear_norm - ear_norm.min()) / (ear_norm.max() - ear_norm.min() + 1e-6)

    return ear_norm.astype(np.float32)


def energy_centroid(R: np.ndarray, percentile: float = 95.0) -> tuple[float, float] | None:
    """
    Compute energy-weighted centroid after percentile thresholding.
    Returns (x_r, y_r) in pixel coordinates, or None if no evidence.
    """
    # threshold high-energy returns
    t = np.percentile(R, percentile)
    R_thr = R - t
    R_thr[R_thr < 0] = 0.0

    denom = float(R_thr.sum())
    if denom <= 1e-9:
        return None

    H, W = R_thr.shape
    ys, xs = np.mgrid[0:H, 0:W]
    x_r = float((xs * R_thr).sum() / denom)
    y_r = float((ys * R_thr).sum() / denom)
    return x_r, y_r


def pick_target_box(boxes_xyxy, policy="largest"):
    """
    If multiple GT boxes exist for the target class in a frame,
    choose one deterministically (default: largest area).
    """
    if not boxes_xyxy:
        return None
    if policy == "first":
        return boxes_xyxy[0]

    # largest area
    best = None
    best_area = -1.0
    for (x1, y1, x2, y2) in boxes_xyxy:
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root directory")
    ap.add_argument("--W", type=int, default=480, help="Width used in your pipeline (default 480)")
    ap.add_argument("--H", type=int, default=640, help="Height used in your pipeline (default 640)")
    ap.add_argument("--percentile", type=float, default=95.0, help="Percentile for energy thresholding")
    ap.add_argument("--target", type=str, default="person_with_knife",
                    choices=["person_with_knife", "person_without_knife", "all"],
                    help="Which GT class to evaluate (after collapse_class_id mapping)")
    ap.add_argument("--out", type=str, default="label_noise_proxy_out", help="Output folder")
    ap.add_argument("--plot", action="store_true", help="Save histogram plot")
    args = ap.parse_args()

    W, H = args.W, args.H
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # class ids after collapse_class_id: 0=without_knife, 1=with_knife :contentReference[oaicite:5]{index=5}
    if args.target == "person_with_knife":
        target_ids = {1}
    elif args.target == "person_without_knife":
        target_ids = {0}
    else:
        target_ids = {0, 1}

    print(f"Scanning items under: {args.root}")
    items = build_items_multi(args.root)
    if not items:
        raise RuntimeError("No items found. Check root path / folder structure.")

    rows = []
    no_evidence = 0
    no_gt = 0

    for idx, (ear_path, lab_path, color_path) in enumerate(items):
        # radar norm map
        R = load_ear_norm_map(ear_path, W, H)
        ctr = energy_centroid(R, percentile=args.percentile)
        if ctr is None:
            no_evidence += 1
            continue
        x_r, y_r = ctr

        # load GT json boxes (camera-space) 
        with open(lab_path, "r") as f:
            data = json.load(f)

        gt_candidates = []
        for b in data.get("boxes", []):
            cid = collapse_class_id(int(b["class_id"]))
            if cid not in target_ids:
                continue
            x1, y1, x2, y2 = map(float, b["bbox_xyxy"])

            # clip to expected bounds (consistent with your exporter) :contentReference[oaicite:7]{index=7}
            x1 = max(0.0, min(x1, W - 1))
            x2 = max(0.0, min(x2, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            y2 = max(0.0, min(y2, H - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            gt_candidates.append((x1, y1, x2, y2))

        gt = pick_target_box(gt_candidates, policy="largest")
        if gt is None:
            no_gt += 1
            continue

        x1, y1, x2, y2 = gt
        x_b = 0.5 * (x1 + x2)
        y_b = 0.5 * (y1 + y2)

        d_px = float(np.sqrt((x_r - x_b) ** 2 + (y_r - y_b) ** 2))
        d_norm = float(d_px / np.sqrt(W * W + H * H))

        bw = float(x2 - x1)
        bh = float(y2 - y1)
        box_diag = float(np.sqrt(bw * bw + bh * bh) + 1e-9)
        d_box = float(d_px / box_diag)

        rows.append({
            "idx": idx,
            "ear_path": ear_path,
            "label_path": lab_path,
            "color_path": color_path,
            "x_r": x_r,
            "y_r": y_r,
            "x_b": x_b,
            "y_b": y_b,
            "d_px": d_px,
            "d_norm": d_norm,
            "d_box": d_box,
            "box_w": bw,
            "box_h": bh,
        })

    # write CSV
    csv_path = out_dir / "per_frame_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary stats
    d_px_arr = np.array([r["d_px"] for r in rows], dtype=float) if rows else np.array([])
    d_box_arr = np.array([r["d_box"] for r in rows], dtype=float) if rows else np.array([])

    def summarize(x: np.ndarray):
        if x.size == 0:
            return {}
        return {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "median": float(np.median(x)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)),
        }

    summary = {
        "N_items_total": len(items),
        "N_used": len(rows),
        "N_no_radar_evidence": int(no_evidence),
        "N_no_target_gt": int(no_gt),
        "W": W,
        "H": H,
        "percentile": float(args.percentile),
        "target": args.target,
        "d_px": summarize(d_px_arr),
        "d_box": summarize(d_box_arr),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DONE ===")
    print(f"Saved per-frame CSV: {csv_path}")
    print(f"Saved summary JSON:  {summary_path}")
    print(f"Used {len(rows)} frames out of {len(items)}")
    print(f"No radar evidence frames: {no_evidence}")
    print(f"No target GT frames:      {no_gt}")

    # optional histogram plot
    if args.plot and d_box_arr.size > 0:
        plt.figure()
        plt.hist(d_box_arr, bins=50)
        plt.xlabel("d_box (centroid-to-box-center distance / box diagonal)")
        plt.ylabel("Count")
        plt.title(f"Cross-modal misalignment proxy (target={args.target})")
        plot_path = out_dir / "hist_d_box.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved histogram plot: {plot_path}")


if __name__ == "__main__":
    main()
