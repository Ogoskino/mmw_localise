"""
eval_heatmap_baseline.py

Evaluate the radar-native heatmap baseline on a dataset split OR on all items.

Produces the same style of safety metrics (precision, recall, F1, MR, FAR)
for the single class 'person_with_knife'.

Examples (PowerShell):

1) Evaluate a split:
  python eval_heatmap_baseline.py `
    --root_dir "C:/.../realsense_data_OD" `
    --ckpt heatmap_runs/run1/best_heatmap_unet.pt `
    --split_json dataset_split.json `
    --split val `
    --thr 0.5 --min_area 50 --iou 0.1

2) Evaluate all items under root_dir (no split needed):
  python eval_heatmap_baseline.py `
    --root_dir "C:/.../test_data" `
    --ckpt heatmap_runs/run1/best_heatmap_unet.pt `
    --all `
    --thr 0.5 --min_area 50 --iou 0.1 `
    --debug --debug_n 25 --debug_out debug_vis
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch

# heatmap_baseline already depends on cv2; we'll use it for drawing/saving.
import cv2

from heatmap_baseline import (
    build_items_multi_ra,
    SmallUNet,
    predict_bboxes,
    evaluate_bbox_metrics,
    load_ra_as_tensor,
    collapse_class_id,
    heatmap_to_bbox,
)


def _load_gt_boxes_xyxy(label_path: str, target_cid: int = 1):
    """Load GT boxes (xyxy) for a target class after collapsing class ids."""
    with open(label_path, "r") as f:
        data = json.load(f)

    gt = []
    for b in data.get("boxes", []):
        cid = collapse_class_id(int(b["class_id"]))
        if cid != target_cid:
            continue
        x1, y1, x2, y2 = b["bbox_xyxy"]
        gt.append([int(x1), int(y1), int(x2), int(y2)])
    return gt


def _draw_boxes(img_bgr: np.ndarray, gt_boxes, pred_box, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """Draw GT (green) and Pred (red) boxes on a BGR image."""
    out = img_bgr.copy()

    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), gt_color, 2)

    if pred_box is not None:
        x1, y1, x2, y2 = int(pred_box.x1), int(pred_box.y1), int(pred_box.x2), int(pred_box.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), pred_color, 2)
        cv2.putText(
            out,
            f"pred:{pred_box.score:.2f}",
            (max(0, x1), max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            pred_color,
            1,
            cv2.LINE_AA,
        )

    return out


def _make_ra_heat_overlay(ra_hw: np.ndarray, heat_hw: np.ndarray):
    """Create a BGR visual: RA grayscale + heatmap colormap overlay."""
    ra_norm = ra_hw.astype(np.float32)
    ra_norm = (ra_norm - ra_norm.min()) / (ra_norm.max() - ra_norm.min() + 1e-6)
    ra_u8 = (ra_norm * 255).astype(np.uint8)
    ra_bgr = cv2.cvtColor(ra_u8, cv2.COLOR_GRAY2BGR)

    hm = np.clip(heat_hw.astype(np.float32), 0.0, 1.0)
    hm_u8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    vis = cv2.addWeighted(ra_bgr, 0.60, hm_color, 0.40, 0)
    return vis


@torch.no_grad()
def _debug_save_examples(
    model: torch.nn.Module,
    eval_items,
    preds,
    device: str,
    out_w: int,
    out_h: int,
    thr: float,
    min_area: int,
    debug_out: str,
    debug_n: int,
    score_thres: float,
    target_cid: int = 1,
):
    """
    Save debug visualisations for first debug_n items:
      - RA+heat overlay with GT+Pred boxes
      - RGB frame with GT+Pred boxes (if available)
    """
    Path(debug_out).mkdir(parents=True, exist_ok=True)

    model.eval()

    n = min(debug_n, len(eval_items))
    for i in range(n):
        ra_path, lab_path, color_path = eval_items[i]

        # Load RA tensor and forward to heatmap prob
        x = load_ra_as_tensor(ra_path, out_w=out_w, out_h=out_h).unsqueeze(0).to(device)  # [1,1,H,W]
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # [H,W]

        # For debug, we optionally recompute pred_box from prob so it's consistent with thr/min_area
        pred_box = heatmap_to_bbox(prob, thr=thr, min_area=min_area)
        if pred_box is not None and pred_box.score < score_thres:
            pred_box = None

        gt_boxes = _load_gt_boxes_xyxy(lab_path, target_cid=target_cid)

        # Load RA raw (for nicer background): use prob map as proxy is ok, but we can also reload .npy
        ra_raw = np.load(ra_path).astype(np.float32)
        ra_raw = np.log(ra_raw + 1e-6)
        ra_raw = cv2.resize(ra_raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        ra_vis = _make_ra_heat_overlay(ra_raw, prob)
        ra_vis = _draw_boxes(ra_vis, gt_boxes, pred_box)

        out_ra = Path(debug_out) / f"dbg_{i:04d}_RA.png"
        cv2.imwrite(str(out_ra), ra_vis)

        # Also draw on the RGB frame if it exists
        if color_path and Path(color_path).exists():
            img = cv2.imread(color_path)
            if img is not None:
                # Ensure RGB image is same size as label coordinate system (your labels are in exported W,H)
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                img_vis = _draw_boxes(img, gt_boxes, pred_box)

                out_rgb = Path(debug_out) / f"dbg_{i:04d}_RGB.png"
                cv2.imwrite(str(out_rgb), img_vis)

    print(f"[debug] Saved {n} examples to: {debug_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="Path to .pt state_dict from training")

    # Make split optional; allow evaluating all items under root_dir.
    ap.add_argument("--split_json", default=None, help="dataset_split.json with indices (optional)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--all", action="store_true", help="Ignore split_json and evaluate all items under root_dir")

    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--H", type=int, default=480)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--thr", type=float, default=0.5, help="Heatmap threshold")
    ap.add_argument("--min_area", type=int, default=50)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score_thres", type=float, default=0.0, help="Min bbox score to keep")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)

    # Debug visuals
    ap.add_argument("--debug", action="store_true", help="Save debug visualisations")
    ap.add_argument("--debug_out", default="debug_vis", help="Folder to save debug images")
    ap.add_argument("--debug_n", type=int, default=20, help="How many frames to save")

    args = ap.parse_args()

    items = build_items_multi_ra(args.root_dir)
    if not items:
        raise RuntimeError(f"No RA items found under: {args.root_dir}")

    # Choose eval set
    if args.all or args.split_json is None:
        eval_items = items
        split_name = "ALL"
    else:
        with open(args.split_json, "r") as f:
            split = json.load(f)

        idx = split.get(args.split, [])
        if not idx:
            raise ValueError(f"Split '{args.split}' is empty in {args.split_json}")

        max_idx = max(idx)
        if max_idx >= len(items):
            raise ValueError(
                f"Split indices out of range for this root_dir.\n"
                f"len(items)={len(items)}, max(split_idx)={max_idx}\n"
                f"This usually means {args.split_json} was generated from a different root_dir "
                f"or different folder/file ordering.\n"
                f"Fix: evaluate with the same root_dir used to generate the split OR pass --all."
            )

        eval_items = [items[i] for i in idx]
        split_name = args.split

    print(f"Evaluating split='{split_name}' with N={len(eval_items)}")

    # Load model
    model = SmallUNet(in_ch=1, base=args.base)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(args.device)

    # Predict bboxes
    preds = predict_bboxes(
        model,
        eval_items,
        device=args.device,
        out_w=args.W,
        out_h=args.H,
        thr=args.thr,
        min_area=args.min_area,
        batch_size=args.batch,
    )

    # Evaluate
    metrics = evaluate_bbox_metrics(
        eval_items,
        preds,
        iou_thres=args.iou,
        score_thres=args.score_thres,
        target_cid=1,
    )

    print("\n===== HEATMAP BASELINE METRICS (person_with_knife) =====")
    print(f"TP/FP/FN/GT: {metrics['TP']} / {metrics['FP']} / {metrics['FN']} / {metrics['GT']}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1:          {metrics['f1']:.4f}")
    print(f"FAR:         {metrics['FAR']:.4f}  (false alarms / frame)")
    print(f"MR:          {metrics['MR']:.4f}  (misses / GT)")
    print("=======================================================\n")

    # Save debug images
    if args.debug:
        _debug_save_examples(
            model=model,
            eval_items=eval_items,
            preds=preds,
            device=args.device,
            out_w=args.W,
            out_h=args.H,
            thr=args.thr,
            min_area=args.min_area,
            debug_out=args.debug_out,
            debug_n=args.debug_n,
            score_thres=args.score_thres,
            target_cid=1,
        )


if __name__ == "__main__":
    main()
