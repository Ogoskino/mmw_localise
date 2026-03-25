# yolo_utils.py

import os
import json
from pathlib import Path

import cv2
import numpy as np

# =========================================================
#  CLASS MAPPING
# =========================================================

YOLO_CLASS_NAMES = {
    0: "person_without_knife",
    1: "person_with_knife",
}

def collapse_class_id(orig_cid):
    """
    Map original class_id (0..3) -> YOLO class index (0 or 1).
    """
    return 0 if orig_cid in (0, 1) else 1


# =========================================================
#  EAR → 3-channel uint8 image
# =========================================================

def load_ear_as_image(ear_path, out_w=480, out_h=640, elev_pool="sum"):
    ear = np.load(ear_path).astype(np.float32)

    if ear.ndim == 3:
        if elev_pool == "mean":
            ear2d = ear.mean(axis=0)
        elif elev_pool == "sum":
            ear2d = ear.sum(axis=0)
        else:
            raise ValueError(f"Unknown elev_pool: {elev_pool}")
    else:
        ear2d = ear

    ear2d = np.log(ear2d + 1e-6)
    ear_resized = cv2.resize(ear2d, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    m = ear_resized.mean()
    s = ear_resized.std() + 1e-6
    ear_norm = (ear_resized - m) / s
    ear_norm = (ear_norm - ear_norm.min()) / (ear_norm.max() - ear_norm.min() + 1e-6)

    img_uint8 = (ear_norm * 255).clip(0, 255).astype(np.uint8)
    img3 = np.stack([img_uint8] * 3, axis=-1)
    return img3



# =========================================================
#  Build (ear, label, color) triples
# =========================================================

def build_items_multi(root_dir, image_exts=(".png", ".jpg", ".jpeg")):
    """
    Recursively search for sessions structured as:

        P_X/
          P_X_S_Y/
            labels_json/
            color_frames_/
            Cascade_Capture_pXX_sYY/
                ear_frames/

    and return a list of (ear_path, label_path, color_path).
    """
    items = []

    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == "ear_frames":
            ear_dir = root
            cascade_dir = os.path.dirname(ear_dir)
            session_dir = os.path.dirname(cascade_dir)

            label_dir = os.path.join(session_dir, "labels_json")
            color_dir = os.path.join(session_dir, "color_frames_")

            if not os.path.isdir(label_dir) or not os.path.isdir(color_dir):
                print(f"⚠️ Skipping {session_dir} – missing labels_json/ or color_frames_/")
                continue

            ear_files = sorted([f for f in os.listdir(ear_dir) if f.endswith(".npy")])
            lab_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
            img_files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(image_exts)])

            n = min(len(ear_files), len(lab_files), len(img_files))
            if n == 0:
                print(f"⚠️ Session {session_dir}: no valid triples")
                continue

            print(f"➡️ Session: {session_dir} | {n} frames")

            for i in range(n):
                items.append((
                    os.path.join(ear_dir, ear_files[i]),
                    os.path.join(label_dir, lab_files[i]),
                    os.path.join(color_dir, img_files[i]),
                ))

    print(f"✅ Total triples: {len(items)}")
    return items


# =========================================================
#  Internal: write a YOLO split (train/val/test)
# =========================================================

def _write_yolo_split(split, items, out_root, W, H):
    """
    Writes images/<split>/*.png and labels/<split>/*.txt.
    Returns list of image paths.
    """
    img_dir = out_root / f"images/{split}"
    lab_dir = out_root / f"labels/{split}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i, (ear_path, lab_path, _) in enumerate(items):
        img = load_ear_as_image(ear_path, W, H)
        name = f"{split}_{i:06d}.png"
        img_path = img_dir / name
        cv2.imwrite(str(img_path), img)

        # Load original boxes
        with open(lab_path, "r") as f:
            data = json.load(f)

        lines = []
        for b in data["boxes"]:
            cid = collapse_class_id(b["class_id"])
            x1, y1, x2, y2 = b["bbox_xyxy"]

            # ---- CLIP TO IMAGE BOUNDS (AVOID >1.0 NORMALIZED) ----
            x1 = max(0.0, min(x1, W - 1))
            x2 = max(0.0, min(x2, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            y2 = max(0.0, min(y2, H - 1))

            if x2 <= x1 or y2 <= y1:
                # degenerate box, skip
                continue

            w = x2 - x1
            h = y2 - y1
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0

            xc_n = xc / W
            yc_n = yc / H
            wn = w / W
            hn = h / H

            # sanity: after clipping these should all be in [0,1]
            xc_n = min(max(xc_n, 0.0), 1.0)
            yc_n = min(max(yc_n, 0.0), 1.0)
            wn   = min(max(wn,   0.0), 1.0)
            hn   = min(max(hn,   0.0), 1.0)

            lines.append(f"{cid} {xc_n:.6f} {yc_n:.6f} {wn:.6f} {hn:.6f}")

        with open(lab_dir / name.replace(".png", ".txt"), "w") as f:
            f.write("\n".join(lines))

        saved_paths.append(str(img_path))

    return saved_paths


# =========================================================
#  Export YOLO dataset (train+val) for TRAINING
# =========================================================

def export_yolo_dataset(train_items, val_items, out_root, W, H):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    val_paths = _write_yolo_split("val",   val_items,   out_root, W, H)
    _          = _write_yolo_split("train", train_items, out_root, W, H)

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
#  Export YOLO dataset (test only) for EVALUATION
# =========================================================

def export_yolo_test_datasets(test_items, out_root, W, H):
    """
    Creates a YOLO-style dataset for the test set:

        out_root/images/test/*.png
        out_root/labels/test/*.txt

    and a YAML where both train and val point to images/test,
    so that YOLO .val() can run on it.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    test_paths = _write_yolo_split("test", test_items, out_root, W, H)

    yaml_path = out_root / "test_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_root.as_posix()}\n")
        f.write("train: images/test\n")
        f.write("val: images/test\n")
        f.write("names:\n")
        for i, nm in YOLO_CLASS_NAMES.items():
            f.write(f"  {i}: {nm}\n")

    return str(yaml_path), test_paths

from pathlib import Path

def export_yolo_test_dataset(
    test_items,
    out_root,
    W,
    H,
    single_class_name=None
):
    """
    Creates a YOLO-style dataset for the test set:

        out_root/images/test/*.png
        out_root/labels/test/*.txt

    If single_class_name is provided:
      - labels are filtered to that class only
      - class indices are NOT remapped
      - YAML remains unchanged (all original classes)
    """

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Write images + labels normally
    test_paths = _write_yolo_split("test", test_items, out_root, W, H)

    # ------------------------------------------------
    # Optional: filter labels to a single class
    # ------------------------------------------------
    if single_class_name is not None:
        name_to_id = {v: k for k, v in YOLO_CLASS_NAMES.items()}
        if single_class_name not in name_to_id:
            raise ValueError(
                f"Class '{single_class_name}' not found in YOLO_CLASS_NAMES: "
                f"{list(YOLO_CLASS_NAMES.values())}"
            )

        target_id = int(name_to_id[single_class_name])
        labels_dir = out_root / "labels" / "test"

        for lbl_path in labels_dir.glob("*.txt"):
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            kept = []
            for L in lines:
                L = L.strip()
                if not L:
                    continue

                parts = L.split()
                cls = int(float(parts[0]))

                if cls == target_id:
                    # KEEP original class id (e.g., 1)
                    kept.append(L + "\n")

            # overwrite label file (may be empty)
            with open(lbl_path, "w") as f:
                f.writelines(kept)

    # ------------------------------------------------
    # Write YAML (unchanged class definitions)
    # ------------------------------------------------
    yaml_path = out_root / "test_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_root.as_posix()}\n")
        f.write("train: images/test\n")
        f.write("val: images/test\n")
        f.write("names:\n")
        for i, nm in YOLO_CLASS_NAMES.items():
            f.write(f"  {i}: {nm}\n")

    return str(yaml_path), test_paths



# =========================================================
#  METRICS (custom TP/FP/FN + FPS)
# =========================================================

def evaluate_metrics(model, items, img_paths, iou_thres=0.5, conf=0.25, verbose=True):
    import time

    num_classes = 2

    TP = [0] * num_classes
    FP = [0] * num_classes
    FN = [0] * num_classes
    GT = [0] * num_classes

    total_frames = len(img_paths)
    inference_times = []

    for (item, img_path) in zip(items, img_paths):
        _, lab_path, _ = item

        with open(lab_path, "r") as f:
            gt_raw = json.load(f)["boxes"]

        gt_boxes = []
        for b in gt_raw:
            cid = collapse_class_id(b["class_id"])
            x1, y1, x2, y2 = b["bbox_xyxy"]
            gt_boxes.append([x1, y1, x2, y2, cid])
            GT[cid] += 1

        # Inference
        t0 = time.time()
        res = model(img_path, conf=conf, verbose=False)[0]
        inference_times.append(time.time() - t0)

        pred_boxes = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            cid = int(box.cls.item())
            sc = float(box.conf.item())
            pred_boxes.append([x1, y1, x2, y2, cid, sc])

        matched_gt = set()

        for px1, py1, px2, py2, pcid, psc in pred_boxes:

            best_iou = 0.0
            best_idx = -1

            for idx_g, (gx1, gy1, gx2, gy2, gcid) in enumerate(gt_boxes):
                if idx_g in matched_gt or gcid != pcid:
                    continue

                ix1 = max(px1, gx1)
                iy1 = max(py1, gy1)
                ix2 = min(px2, gx2)
                iy2 = min(py2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_p = (px2 - px1) * (py2 - py1)
                area_g = (gx2 - gx1) * (gy2 - gy1)
                iou = inter / (area_p + area_g - inter + 1e-6)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx_g

            if best_iou >= iou_thres:
                TP[pcid] += 1
                matched_gt.add(best_idx)
            else:
                FP[pcid] += 1

        # Missed GT
        for idx_g, (_, _, _, _, gcid) in enumerate(gt_boxes):
            if idx_g not in matched_gt:
                FN[gcid] += 1

    # Per-class metrics
    per_class = {}
    for cid in range(num_classes):
        tp, fp, fn, gt = TP[cid], FP[cid], FN[cid], GT[cid]
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        FAR = fp / total_frames
        MR = fn / (gt + 1e-6)

        per_class[YOLO_CLASS_NAMES[cid]] = {
            "TP": tp, "FP": fp, "FN": fn, "GT": gt,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "FAR": FAR,
            "MR": MR,
        }

    # Global metrics
    TP_T = sum(TP)
    FP_T = sum(FP)
    FN_T = sum(FN)
    GT_T = sum(GT)

    global_precision = TP_T / (TP_T + FP_T + 1e-6)
    global_recall = TP_T / (TP_T + FN_T + 1e-6)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-6)
    global_FAR = FP_T / total_frames
    global_MR = FN_T / (GT_T + 1e-6)
    FPS = 1.0 / (sum(inference_times) / len(inference_times))

    if verbose:
        print("\n========== PER-CLASS METRICS ==========")
        for cname, m in per_class.items():
            print(f"\nClass: {cname}")
            print(f" TP/FP/FN/GT = {m['TP']} / {m['FP']} / {m['FN']} / {m['GT']}")
            print(f" Precision:  {m['precision']:.4f}")
            print(f" Recall:     {m['recall']:.4f}")
            print(f" F1:         {m['f1']:.4f}")
            print(f" FAR:        {m['FAR']:.4f}")
            print(f" MR:         {m['MR']:.4f}")

        print("\n------------ GLOBAL METRICS ------------")
        print(f" TP/FP/FN/GT = {TP_T} / {FP_T} / {FN_T} / {GT_T}")
        print(f" Precision:  {global_precision:.4f}")
        print(f" Recall:     {global_recall:.4f}")
        print(f" F1 Score:   {global_f1:.4f}")
        print(f" FAR:        {global_FAR:.4f}")
        print(f" MR:         {global_MR:.4f}")
        print(f" FPS:        {FPS:.2f}")
        print("========================================\n")

    return {
        "per_class": per_class,
        "global": {
            "precision": global_precision,
            "recall": global_recall,
            "f1": global_f1,
            "FAR": global_FAR,
            "MR": global_MR,
            "FPS": FPS
        }
    }


# =========================================================
#  VISUALISATION (GT + predictions)
# =========================================================

def visualize_predictions(model, items, img_paths, conf=0.25, out_dir="vis_output", K=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ((_, lab_path, color_path), pred_img) in enumerate(zip(items, img_paths)):
        if i >= K:
            break

        color = cv2.imread(color_path)
        if color is None:
            print(f"⚠️ Could not read color image: {color_path}")
            continue

        with open(lab_path, "r") as f:
            gt = json.load(f)["boxes"]

        vis = color.copy()

        # --------------------------------------------------
        #  GT BOXES (Green) — LABEL ABOVE BOX
        # --------------------------------------------------
        for b in gt:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            cid = collapse_class_id(b["class_id"])
            nm = YOLO_CLASS_NAMES[cid]

            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"GT:{nm}",
                (int(x1), int(y1) - 5),   # ABOVE GT BOX
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # --------------------------------------------------
        #  PREDICTIONS (Red) — LABEL *BELOW* BOX
        # --------------------------------------------------
        results = model(pred_img, conf=conf, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            cid = int(box.cls.item())
            sc = float(box.conf.item())
            nm = YOLO_CLASS_NAMES[cid]

            if nm == "person_without_knife":
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # BELOW predicted box
                text_y = int(y2) + 15  
                if text_y > vis.shape[0] - 5:
                    text_y = vis.shape[0] - 5  # keep inside image

                cv2.putText(
                    vis,
                    f"P:{nm} {sc:.2f}",
                    (int(x1), text_y),  # BELOW PRED BOX
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
            else:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # BELOW predicted box
                text_y = int(y2) + 15  
                if text_y > vis.shape[0] - 5:
                    text_y = vis.shape[0] - 5  # keep inside image

                cv2.putText(
                    vis,
                    f"P:{nm} {sc:.2f}",
                    (int(x1), text_y),  # BELOW PRED BOX
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )

        cv2.imwrite(str(out_dir / f"sample_{i}.jpg"), vis)

    print(f"Saved visualisation samples to {out_dir}")



# =========================================================
#  YOLO mAP EVALUATION
# =========================================================

def evaluate_map_yolo(model, yaml_path, split="val"):
    """
    Uses Ultralytics built-in evaluator to compute:
    - AP50  -> metrics/mAP50(B)
    - AP    -> metrics/mAP50-95(B)
    """
    results = model.val(data=yaml_path, split=split, verbose=False)
    ap50 = results.results_dict["metrics/mAP50(B)"]
    ap   = results.results_dict["metrics/mAP50-95(B)"]
    return ap50, ap

# =========================================================
#  BEST OPERATING POINT SEARCH (ON VAL SET)
# =========================================================

def find_best_operating_point(model, val_items, val_paths):
    """
    Scan conf in [0.05..0.95] at IoU=0.5 and choose the
    best score for the `person_with_knife` class.

    Score = (1-MR)^4 * (1-FAR)^2 * (F1)^1
    """
    conf_list = np.linspace(0.01, 0.5, 50)
    iou = 0.5

    best_score = -1.0
    best_conf = None
    best_metrics = None

    for conf in conf_list:
        metrics = evaluate_metrics(
            model,
            val_items,
            val_paths,
            iou_thres=iou,
            conf=float(conf),
            verbose=False
        )

        knife = metrics["per_class"]["person_with_knife"]
        F1  = knife["f1"]
        MR  = knife["MR"]
        FAR = knife["FAR"]

        score = ((1 - MR)) * ((1 - FAR)) * (F1)

        if score > best_score:
            best_score = score
            best_conf = conf
            best_metrics = knife

    print("\n===== Global Best Operating Point (val set) =====")
    print(f"Best Score = {best_score:.6f}")
    print(f"Best conf  = {best_conf:.2f}")
    print(f"MR={best_metrics['MR']:.3f}, FAR={best_metrics['FAR']:.3f}, F1={best_metrics['f1']:.3f}")

    return {
        "best_conf": best_conf,
        "best_score": best_score,
        "best_metrics": best_metrics,
    }


import numpy as np
import random
import time

def bootstrap_metrics(
    model,
    test_items,
    test_img_paths,
    evaluate_metrics_fn,
    iou_thres=0.50,
    conf=0.30,
    n_boot=500,
    seed=0,
    class_name="person_with_knife",
    verbose=False
):
    """
    Bootstraps MR/FAR/F1 over test samples (frames).
    Returns mean + 95% CI for the selected class.
    """

    rng = np.random.default_rng(seed)
    N = len(test_items)

    mr_list, far_list, f1_list = [], [], []

    t0 = time.time()

    for b in range(n_boot):
        # sample indices with replacement
        idxs = rng.integers(low=0, high=N, size=N)

        boot_items = [test_items[i] for i in idxs]
        boot_paths = [test_img_paths[i] for i in idxs]

        metrics = evaluate_metrics_fn(
            model,
            boot_items,
            boot_paths,
            iou_thres=iou_thres,
            conf=conf,
            verbose=False
        )

        pc = metrics["per_class"][class_name]
        mr_list.append(pc["MR"])
        far_list.append(pc["FAR"])
        f1_list.append(pc["f1"])

        if verbose and (b + 1) % 50 == 0:
            print(f"Bootstrap {b+1}/{n_boot}")

    mr_arr = np.array(mr_list, dtype=float)
    far_arr = np.array(far_list, dtype=float)
    f1_arr = np.array(f1_list, dtype=float)

    def summarize(x):
        mean = float(np.mean(x))
        lo = float(np.percentile(x, 2.5))
        hi = float(np.percentile(x, 97.5))
        return mean, lo, hi

    out = {
        "MR": summarize(mr_arr),
        "FAR": summarize(far_arr),
        "F1": summarize(f1_arr),
        "n_boot": n_boot,
        "N": N,
        "time_sec": time.time() - t0
    }
    return out