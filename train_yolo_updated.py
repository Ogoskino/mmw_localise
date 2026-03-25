"""
TRAINING SCRIPT FOR YOLO MODELS USING MMWAVE RADAR DATA
Uses yolo_utils.py for all utility functions.
"""

import json
from ultralytics import YOLO

from yolo_utils import (
    build_items_multi,
    export_yolo_dataset,
    evaluate_metrics,
    find_best_operating_point,
    visualize_predictions,
)

# =========================================================
# CONFIGURATION
# =========================================================

W, H = 480, 640

MODEL_NAMES = ["yolov8", "yolov9", "yolov10", "yolov11", "yolov12"]
PRETRAIN_WEIGHTS = [
    "yolov8n.pt",
    "yolov9t.pt",
    "yolov10n.pt",
    "yolo11n.pt",
    "yolo12n.pt"
]

OUT_PROJECT = "yolo_radar_runs"


# =========================================================
# TRAIN PIPELINE
# =========================================================

def train_yolo(root_dir):

    # -----------------------------------------------------
    # 1. Build training + validation datasets
    # -----------------------------------------------------
    print("\nScanning dataset...")
    items = build_items_multi(root_dir)

    if len(items) == 0:
        raise RuntimeError("❌ ERROR: No samples found in dataset directory!")

    with open("dataset_split.json", "r") as f:
        split_data = json.load(f)

    train_items = [items[i] for i in split_data["train"]]
    val_items   = [items[i] for i in split_data["val"]]

    print(f"\nUsing unified split → {len(train_items)} train, {len(val_items)} val samples.")

    yaml_path, val_paths = export_yolo_dataset(train_items, val_items,
                                               out_root="yolo_radar_ds",
                                               W=W, H=H)

    # -----------------------------------------------------
    # 2. Train each YOLO model
    # -----------------------------------------------------
    model_summaries = []

    for name, pretrained in zip(MODEL_NAMES, PRETRAIN_WEIGHTS):

        print("\n====================================================")
        print(f" TRAINING MODEL → {name}")
        print("====================================================\n")

        model = YOLO(pretrained)

        # train model
        model.train(
            data=yaml_path,
            epochs=180,
            imgsz=H,
            batch=16,
            device=0,
            project=OUT_PROJECT,
            name=name,
            exist_ok=True
        )

        # -------------------------------------------------
        # 3. Evaluate  → find best confidence threshold
        # -------------------------------------------------
        print("\nSearching for best operating point...")
        best = find_best_operating_point(model, val_items, val_paths)

        best_conf = best["best_conf"]
        best_metrics = best["best_metrics"]

        # -------------------------------------------------
        # 4. Visualise predictions
        # -------------------------------------------------
        visualize_predictions(
            model, val_items, val_paths,
            conf=best_conf,
            out_dir=f"best_visualisations/{name}",
            K=10
        )

        # -------------------------------------------------
        # 5. Store comparison row
        # -------------------------------------------------
        model_summaries.append({
            "model": name,
            "conf": best_conf,
            "score": best["best_score"],
            "MR": best_metrics["MR"],
            "FAR": best_metrics["FAR"],
            "F1": best_metrics["f1"]
        })

    # -----------------------------------------------------
    # 6. Print comparison table
    # -----------------------------------------------------
    print("\n\n================ MODEL COMPARISON ==================\n")
    header = f"{'Model':<12} {'Conf':<6} {'Score':<12} {'MR':<8} {'FAR':<8} {'F1':<8}"
    print(header)
    print("-" * len(header))

    for row in model_summaries:
        print(f"{row['model']:<12}"
              f"{row['conf']:<6.2f}"
              f"{row['score']:<12.4f}"
              f"{row['MR']:<8.3f}"
              f"{row['FAR']:<8.3f}"
              f"{row['F1']:<8.3f}")

    print("\n====================================================")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    root_dir = r"C:\Users\n1071552\Desktop\projects\data_collectn\realsense_data_OD"
    train_yolo(root_dir)
