"""heatmap_baseline.py

Radar-native learned baseline:
  RA map (range-azimuth) -> 1-channel heatmap -> bbox via connected components.

Designed to plug into the existing project structure where:
  session_dir/
    labels_json/*.json
    color_frames_/*.(png|jpg|jpeg)
    Cascade_Capture_*/
      ra_frames/*.npy

The exported bbox coordinate system matches the same (W,H) used in YOLO export
(default 480x640), i.e., *camera-space* boxes that were clipped/resized to the
export resolution.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# -------------------------------
# Class mapping (keep consistent)
# -------------------------------

YOLO_CLASS_NAMES = {
    0: "person_without_knife",
    1: "person_with_knife",
}


def collapse_class_id(orig_cid: int) -> int:
    """Map original class_id (0..3) -> binary class (0/1)."""
    return 0 if orig_cid in (0, 1) else 1


# -------------------------------
# Data discovery
# -------------------------------

def build_items_multi_ra(root_dir: str, image_exts=(".png", ".jpg", ".jpeg")):
    """Return list of (ra_path, label_path, color_path).

    Mirrors yolo_utils.build_items_multi() but searches for 'ra_frames' instead
    of 'ear_frames'.
    """
    items = []

    # Ensure deterministic traversal order (important when using a precomputed
    # split file containing indices).
    for root, dirs, _ in os.walk(root_dir, topdown=True):
        dirs.sort()
        if os.path.basename(root) == "ra_frames":
            ra_dir = root
            cascade_dir = os.path.dirname(ra_dir)
            session_dir = os.path.dirname(cascade_dir)

            label_dir = os.path.join(session_dir, "labels_json")
            color_dir = os.path.join(session_dir, "color_frames_")

            if not os.path.isdir(label_dir) or not os.path.isdir(color_dir):
                # keep quiet; calling script can print summary
                continue

            ra_files = sorted([f for f in os.listdir(ra_dir) if f.endswith(".npy")])
            lab_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
            img_files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(image_exts)])

            n = min(len(ra_files), len(lab_files), len(img_files))
            if n == 0:
                continue

            for i in range(n):
                items.append(
                    (
                        os.path.join(ra_dir, ra_files[i]),
                        os.path.join(label_dir, lab_files[i]),
                        os.path.join(color_dir, img_files[i]),
                    )
                )

    return items


# -------------------------------
# RA preprocessing
# -------------------------------

def load_ra_as_tensor(ra_path: str, out_w: int = 480, out_h: int = 640) -> torch.Tensor:
    """Load RA .npy (float32) and convert to normalized torch tensor [1,H,W]."""
    ra = np.load(ra_path).astype(np.float32)  # expected roughly in [0,1]

    # Stabilize dynamic range a bit
    ra = np.log(ra + 1e-6)

    ra_resized = cv2.resize(ra, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    m = float(ra_resized.mean())
    s = float(ra_resized.std() + 1e-6)
    ra_norm = (ra_resized - m) / s
    ra_norm = (ra_norm - ra_norm.min()) / (ra_norm.max() - ra_norm.min() + 1e-6)

    x = torch.from_numpy(ra_norm).float().unsqueeze(0)  # [1,H,W]
    return x


def boxes_to_mask(label_path: str, out_w: int, out_h: int, target_cid: int = 1) -> torch.Tensor:
    """Create a binary mask [1,H,W] from GT boxes of a target class.

    We use collapse_class_id() and keep only target_cid.
    """
    with open(label_path, "r") as f:
        data = json.load(f)

    mask = np.zeros((out_h, out_w), dtype=np.float32)
    for b in data.get("boxes", []):
        cid = collapse_class_id(int(b["class_id"]))
        if cid != target_cid:
            continue
        x1, y1, x2, y2 = b["bbox_xyxy"]
        # clip
        x1 = int(max(0, min(out_w - 1, x1)))
        x2 = int(max(0, min(out_w - 1, x2)))
        y1 = int(max(0, min(out_h - 1, y1)))
        y2 = int(max(0, min(out_h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 1.0

    return torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]


class RADataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, str, str]],
        out_w: int = 480,
        out_h: int = 640,
        target_cid: int = 1,
    ):
        self.items = items
        self.out_w = out_w
        self.out_h = out_h
        self.target_cid = target_cid

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ra_path, lab_path, _ = self.items[idx]
        x = load_ra_as_tensor(ra_path, self.out_w, self.out_h)
        y = boxes_to_mask(lab_path, self.out_w, self.out_h, target_cid=self.target_cid)
        return x, y


# -------------------------------
# Model: small UNet-ish
# -------------------------------

def _conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Small UNet-like network for heatmap prediction."""

    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = _conv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _conv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _conv(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv(base * 2, base)

        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)  # logits [B,1,H,W]


# -------------------------------
# Inference: heatmap -> bbox
# -------------------------------

@dataclass
class BBoxPred:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def heatmap_to_bbox(prob_map: np.ndarray, thr: float = 0.5, min_area: int = 50) -> Optional[BBoxPred]:
    """Convert a probability map [H,W] to a single bbox via connected components."""
    H, W = prob_map.shape
    bin_map = (prob_map >= thr).astype(np.uint8)

    if bin_map.max() == 0:
        return None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
    # stats: [label, x, y, w, h, area]
    # label 0 is background
    best = None
    best_area = 0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, w, h, area)

    if best is None:
        return None

    x, y, w, h, area = best
    x1 = float(x)
    y1 = float(y)
    x2 = float(min(W - 1, x + w))
    y2 = float(min(H - 1, y + h))

    # score: mean prob inside bbox
    crop = prob_map[int(y1):int(y2), int(x1):int(x2)]
    score = float(crop.mean()) if crop.size else float(prob_map.max())
    return BBoxPred(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


@torch.no_grad()
def predict_bboxes(
    model: nn.Module,
    items: List[Tuple[str, str, str]],
    device: str = "cuda",
    out_w: int = 480,
    out_h: int = 640,
    thr: float = 0.5,
    min_area: int = 50,
    batch_size: int = 4,
) -> List[Optional[BBoxPred]]:
    """Run model on items and return one bbox per frame (or None)."""
    model.eval()
    preds: List[Optional[BBoxPred]] = []

    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        xs = [load_ra_as_tensor(p, out_w, out_h) for (p, _, _) in chunk]
        x = torch.stack(xs, dim=0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # [B,H,W]

        for pm in probs:
            preds.append(heatmap_to_bbox(pm, thr=thr, min_area=min_area))

    return preds


def evaluate_bbox_metrics(
    items: List[Tuple[str, str, str]],
    preds: List[Optional[BBoxPred]],
    iou_thres: float = 0.5,
    score_thres: float = 0.0,
    target_cid: int = 1,
) -> Dict:
    """Evaluate single-class detection (person_with_knife) using IoU matching.

    FAR is false alarms per frame, MR is miss rate per GT instances.
    """
    assert len(items) == len(preds)

    TP = FP = FN = GT = 0
    total_frames = len(items)

    for (ra_path, lab_path, _), pred in zip(items, preds):
        with open(lab_path, "r") as f:
            gt_raw = json.load(f).get("boxes", [])

        gt_boxes = []
        for b in gt_raw:
            cid = collapse_class_id(int(b["class_id"]))
            if cid != target_cid:
                continue
            x1, y1, x2, y2 = b["bbox_xyxy"]
            gt_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        GT += len(gt_boxes)

        # Prediction decision
        if pred is None or pred.score < score_thres:
            # no prediction
            FN += len(gt_boxes)
            continue

        # If multiple GT boxes, match best IoU (rare here)
        best_iou = 0.0
        for g in gt_boxes:
            best_iou = max(best_iou, iou_xyxy([pred.x1, pred.y1, pred.x2, pred.y2], g))

        if len(gt_boxes) == 0:
            FP += 1
        else:
            if best_iou >= iou_thres:
                TP += 1
                # count other GT as missed (if any)
                FN += max(0, len(gt_boxes) - 1)
            else:
                FP += 1
                FN += len(gt_boxes)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    FAR = FP / (total_frames + 1e-6)
    MR = FN / (GT + 1e-6)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "GT": GT,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "FAR": FAR,
        "MR": MR,
    }


# -------------------------------
# Training utilities
# -------------------------------


def compute_pos_weight(dataset: Dataset, max_samples: int = 200) -> float:
    """Estimate a pos_weight for BCE to counter class imbalance.

    We sample up to max_samples masks to estimate foreground fraction.
    pos_weight = (neg / pos). If pos is tiny, cap to a reasonable value.
    """
    n = min(len(dataset), max_samples)
    if n <= 0:
        return 1.0

    pos = 0.0
    tot = 0.0
    step = max(1, len(dataset) // n)
    for i in range(0, len(dataset), step):
        _, y = dataset[i]
        y_np = y.numpy()
        pos += float((y_np > 0.5).sum())
        tot += float(y_np.size)
        if (tot / y_np.size) >= n:
            break

    neg = max(1.0, tot - pos)
    pos = max(1.0, pos)
    pw = neg / pos
    return float(min(max(pw, 1.0), 50.0))


def train_heatmap_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Optional[Dataset] = None,
    device: str = "cuda",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-3,
    pos_weight: Optional[float] = None,
    num_workers: int = 0,
    save_path: Optional[str] = None,
):
    """Train the heatmap model with BCEWithLogitsLoss.

    Returns a dict with training history and best val loss (if val_ds).
    """
    from torch.utils.data import DataLoader

    model = model.to(device)

    if pos_weight is None:
        pos_weight = compute_pos_weight(train_ds)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            vr = 0.0
            vn = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    vr += float(loss.item())
                    vn += 1
            val_loss = vr / max(1, vn)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if save_path is not None:
                    torch.save(best_state, save_path)

        # minimal logging
        if val_loss is None:
            print(f"[ep {ep:03d}] train_loss={train_loss:.4f} pos_weight={pos_weight:.2f}")
        else:
            print(f"[ep {ep:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} pos_weight={pos_weight:.2f}")

    if best_state is not None and save_path is None:
        # restore best into model
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_val_loss": best_val if val_ds is not None else None,
        "pos_weight": pos_weight,
    }
