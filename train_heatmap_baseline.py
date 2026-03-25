"""train_heatmap_baseline.py

Train the radar-native heatmap baseline (SmallUNet) on RA maps.

This script mirrors your YOLO training pipeline but uses:
  RA .npy -> 1-channel heatmap supervision from GT bboxes.

Usage (example):
  python train_heatmap_baseline.py \
    --root_dir "C:/.../realsense_data_OD" \
    --split_json dataset_split.json \
    --out_dir heatmap_runs/run1 \
    --epochs 40 --batch 6 --lr 1e-3 \
    --W 480 --H 640
"""

import os
import json
import argparse
from pathlib import Path

import torch

from heatmap_baseline import (
    build_items_multi_ra,
    RADataset,
    SmallUNet,
    train_heatmap_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Dataset root containing session folders")
    ap.add_argument("--split_json", required=True, help="dataset_split.json produced by generate_split.py")
    ap.add_argument("--out_dir", required=True, help="Output directory for checkpoints/logs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--H", type=int, default=480)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base", type=int, default=32, help="UNet base channels")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning dataset for ra_frames...")
    items = build_items_multi_ra(args.root_dir)
    if not items:
        raise RuntimeError(f"No RA items found under: {args.root_dir}")
    print(f"Found {len(items)} paired (ra,label,color) items")

    with open(args.split_json, "r") as f:
        split = json.load(f)
    train_idx = split.get("train", [])
    val_idx = split.get("val", [])

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    print(f"Train items: {len(train_items)} | Val items: {len(val_items)}")

    train_ds = RADataset(train_items, out_w=args.W, out_h=args.H, target_cid=1)
    val_ds = RADataset(val_items, out_w=args.W, out_h=args.H, target_cid=1) if val_items else None

    model = SmallUNet(in_ch=1, base=args.base)
    ckpt_path = str(out_dir / "best_heatmap_unet.pt")

    print(f"Training on device={args.device} | saving to {ckpt_path}")
    info = train_heatmap_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        num_workers=args.num_workers,
        save_path=ckpt_path,
    )

    # save training meta
    meta = {
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "W": args.W,
        "H": args.H,
        "base": args.base,
        "device": args.device,
        "best_val_loss": info.get("best_val_loss"),
        "pos_weight": info.get("pos_weight"),
    }
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
