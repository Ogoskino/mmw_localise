# test_evaluate.py

import os
import multiprocessing as mp
from ultralytics import YOLO

from yolo_utils import (
    build_items_multi,
    export_yolo_test_dataset,
    evaluate_metrics,
    evaluate_map_yolo,
    visualize_predictions,
    bootstrap_metrics,
)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

TEST_ROOT_DIR = r"C:\Users\n1071552\Desktop\projects\data_collectn\test_data"
TEST_OUT_ROOT = "yolo_radar_test_ds"

W, H = 480, 640

MODEL_PATHS = {
    "yolov8":  "yolo_radar_runs/yolov8/weights/best.pt",
    "yolov9":  "yolo_radar_runs/yolov9/weights/best.pt",
    "yolov10": "yolo_radar_runs/yolov10/weights/best.pt",
    "yolov11": "yolo_radar_runs/yolov11/weights/best.pt",
    "yolov12": "yolo_radar_runs/yolov12/weights/best.pt",
}

CONF_THRESHOLDS = {
    "yolov8":  0.37,
    "yolov9":  0.27,
    "yolov10": 0.34,
    "yolov11": 0.43,
    "yolov12": 0.38,
}

IOU_THRESHOLD = 0.50


def fmt_ci(ci):
    mean, lo, hi = ci
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]"



def main():

    # ---------------------------------------------------
    # 1. Build test items
    # ---------------------------------------------------
    print("\nScanning test dataset...")
    test_items = build_items_multi(TEST_ROOT_DIR)
    if len(test_items) == 0:
        raise RuntimeError("ERROR: No test samples found!")

    print(f"Total test samples: {len(test_items)}")

    # ---------------------------------------------------
    # 2. Export YOLO-style dataset
    # ---------------------------------------------------
    print("\nExporting YOLO-format test dataset...")
    yaml_test_path, test_img_paths = export_yolo_test_dataset(
        test_items,
        out_root=TEST_OUT_ROOT,
        W=W,
        H=H,
        single_class_name="person_with_knife"
    )

    print(f"Test YAML: {yaml_test_path}")
    print(f"Total test images saved: {len(test_img_paths)}")

    # ---------------------------------------------------
    # 3. Evaluate all models
    # ---------------------------------------------------
    results_table = []

    for model_key, model_path in MODEL_PATHS.items():
        print("\n" + "=" * 70)
        print(f"Evaluating model: {model_key}")
        print("=" * 70)

        conf_thres = CONF_THRESHOLDS[model_key]
        print(f"Using CONF={conf_thres}, IOU={IOU_THRESHOLD}")

        model = YOLO(model_path)

        # ---------------------------
        # AP metrics (YOLO native)
        # ---------------------------
        print("Running YOLO AP evaluation...")
        ap50, ap = evaluate_map_yolo(model, yaml_test_path, split="val")

        # ---------------------------
        # Deterministic metrics
        # ---------------------------
        print("Running custom safety metrics...")
        metrics = evaluate_metrics(
            model,
            test_items,
            test_img_paths,
            iou_thres=IOU_THRESHOLD,
            conf=conf_thres,
            verbose=False
        )

        per_class = metrics["per_class"]["person_with_knife"]
        fps = metrics["global"]["FPS"]

        MR = per_class["MR"]
        FAR = per_class["FAR"]
        F1 = per_class["f1"]

        # ---------------------------
        # Bootstrap uncertainty
        # ---------------------------
        print("Running bootstrap uncertainty estimation...")
        boot = bootstrap_metrics(
            model=model,
            test_items=test_items,
            test_img_paths=test_img_paths,
            evaluate_metrics_fn=evaluate_metrics,
            iou_thres=IOU_THRESHOLD,
            conf=conf_thres,
            n_boot=1000,
            seed=42,
            class_name="person_with_knife",
            verbose=True
        )

        # ---------------------------
        # Store results (ONE row)
        # ---------------------------
        results_table.append({
            "model": model_key,
            "conf": conf_thres,
            "AP50": ap50,
            "AP": ap,
            "MR": MR,
            "FAR": FAR,
            "F1": F1,
            "MR_CI": boot["MR"],
            "FAR_CI": boot["FAR"],
            "F1_CI": boot["F1"],
            "FPS": fps,
            "N": boot["N"],
        })

        # ---------------------------
        # Visualisation
        # ---------------------------
        vis_dir = os.path.join("test_visualisations", model_key)
        visualize_predictions(
            model,
            test_items,
            test_img_paths,
            conf=conf_thres,
            out_dir=vis_dir,
            K=1000
        )

    # ---------------------------------------------------
    # 4. Final Table
    # ---------------------------------------------------
    print("\n\n====================== FINAL MODEL COMPARISON ======================")
    header = (
        f"{'Model':<10} {'Conf':<6} {'AP50':<8} {'AP':<8} "
        f"{'MR (95% CI)':<22} {'FAR (95% CI)':<22} {'F1 (95% CI)':<22} {'FPS':<6}"
    )
    print(header)
    print("-" * len(header))

    for row in results_table:
        print(
            f"{row['model']:<10} "
            f"{row['conf']:<6.2f} "
            f"{row['AP50']:<8.3f} "
            f"{row['AP']:<8.3f} "
            f"{fmt_ci(row['MR_CI']):<22} "
            f"{fmt_ci(row['FAR_CI']):<22} "
            f"{fmt_ci(row['F1_CI']):<22} "

            f"{row['FPS']:<6.1f}"
        )

    print("\n====================================================================")


if __name__ == "__main__":
    mp.freeze_support()
    main()
