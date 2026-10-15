"""
Run Inference using trained MobileNetV2 to predict Kurta Fabric / Material.
Supports batch directory processing, confidence scores, safe renaming, copy mode, and Excel/CSV reports.
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import pandas as pd

import config


def load_model(model_path: Path, num_classes: int, device: torch.device):
    """Loads trained MobileNetV2 model architecture and state dictionary."""
    model = torchvision.models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_class_labels(model_path: Path) -> List[str]:
    """Retrieves class names from classes.json if present, otherwise defaults to config.CLASSES."""
    classes_json = model_path.parent / "classes.json"
    if classes_json.exists():
        try:
            with open(classes_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "classes" in data and isinstance(data["classes"], list):
                    return data["classes"]
        except Exception as e:
            print(f"[WARN] Failed to read classes.json ({e}). Falling back to config.CLASSES.")
    return config.CLASSES


def predict_single_image(model, image_path: Path, transform, device: torch.device, classes: List[str]) -> Tuple[str, float]:
    """Runs inference on a single image and returns (predicted_class, confidence_percentage)."""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_label = classes[pred_idx.item()]
    conf_score = confidence.item() * 100.0
    return predicted_label, conf_score


def run_batch_prediction(
    input_path: str,
    model_path: str = str(config.DEFAULT_MODEL_PATH),
    excel_path: str = str(config.DEFAULT_RESULTS_EXCEL),
    csv_path: str = str(config.DEFAULT_RESULTS_CSV),
    action: str = "none",
    output_dir: str = None,
):
    """
    Executes prediction pipeline over an image or folder of images.
    
    Actions:
      - 'none': Analyze and report only (keeps original files untouched).
      - 'rename': Renames files in-place (e.g., cotton_1.jpg).
      - 'copy': Copies files to categorized folders (e.g., output/cotton/...).
    """
    device = config.DEVICE
    model_p = Path(model_path)
    input_p = Path(input_path)

    if not input_p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    classes = get_class_labels(model_p)
    print(f"[INFO] Using {len(classes)} classes: {classes}")
    print(f"[INFO] Loading model from: {model_p}")

    model = load_model(model_p, len(classes), device)

    # Collect image files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in valid_exts else []
    else:
        image_files = [p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]

    if not image_files:
        print(f"[WARN] No valid images found in {input_p}")
        return

    print(f"[INFO] Processing {len(image_files)} image(s)...")

    results = []
    category_counts = {c: 0 for c in classes}
    file_counters = {}

    for idx, img_path in enumerate(image_files, 1):
        try:
            pred_class, confidence = predict_single_image(
                model, img_path, config.INFERENCE_TRANSFORMS, device, classes
            )
            category_counts[pred_class] += 1
            file_counters[pred_class] = file_counters.get(pred_class, 0) + 1

            new_name = f"{pred_class}_{file_counters[pred_class]}{img_path.suffix.lower()}"
            status = "analyzed"

            if action == "rename":
                target_path = img_path.parent / new_name
                # Avoid collision with existing files
                while target_path.exists() and target_path != img_path:
                    file_counters[pred_class] += 1
                    new_name = f"{pred_class}_{file_counters[pred_class]}{img_path.suffix.lower()}"
                    target_path = img_path.parent / new_name

                if target_path != img_path:
                    img_path.rename(target_path)
                    status = f"renamed -> {new_name}"

            elif action == "copy":
                destination_folder = (Path(output_dir) if output_dir else img_path.parent / "categorized") / pred_class
                destination_folder.mkdir(parents=True, exist_ok=True)
                dest_path = destination_folder / new_name
                shutil.copy2(img_path, dest_path)
                status = f"copied -> {destination_folder.name}/{new_name}"

            results.append({
                "Original_File": img_path.name,
                "Predicted_Material": pred_class,
                "Confidence (%)": round(confidence, 2),
                "Assigned_Name": new_name,
                "Action_Status": status,
                "Source_Path": str(img_path)
            })

            print(f"[{idx:03d}/{len(image_files):03d}] {img_path.name} -> {pred_class} ({confidence:.1f}%) [{status}]")

        except Exception as e:
            print(f"[ERROR] Failed processing {img_path.name}: {e}")
            results.append({
                "Original_File": img_path.name,
                "Predicted_Material": "ERROR",
                "Confidence (%)": 0.0,
                "Assigned_Name": img_path.name,
                "Action_Status": f"Failed: {e}",
                "Source_Path": str(img_path)
            })

    # Save Reports
    df = pd.DataFrame(results)

    # Export to Excel
    excel_saved = False
    try:
        df.to_excel(excel_path, index=False)
        print(f"\n[SUCCESS] Excel report saved to: {excel_path}")
        excel_saved = True
    except Exception as e:
        print(f"[WARN] Could not write Excel file ({e}). Saving CSV backup.")

    # Export to CSV backup
    try:
        df.to_csv(csv_path, index=False)
        print(f"[SUCCESS] CSV report saved to: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Could not write CSV report: {e}")

    # Summary
    print("\n" + "=" * 45)
    print("Prediction Summary")
    print("=" * 45)
    for cat, count in category_counts.items():
        print(f" - {cat.capitalize():<10}: {count} images")
    print(f"Total Processed: {len(results)}")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Kurta Fabric Material using MobileNetV2")
    parser.add_argument("--input", type=str, default=str(config.DATASET_DIR), help="Path to input image or directory")
    parser.add_argument("--model", type=str, default=str(config.DEFAULT_MODEL_PATH), help="Path to model weights (.pth)")
    parser.add_argument("--excel", type=str, default=str(config.DEFAULT_RESULTS_EXCEL), help="Path for output Excel report")
    parser.add_argument("--csv", type=str, default=str(config.DEFAULT_RESULTS_CSV), help="Path for output CSV report")
    parser.add_argument("--action", type=str, choices=["none", "rename", "copy"], default="none",
                        help="Action to perform on images: 'none' (report only), 'rename' (in-place), or 'copy'")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination directory if action is 'copy'")

    args = parser.parse_args()
    run_batch_prediction(
        input_path=args.input,
        model_path=args.model,
        excel_path=args.excel,
        csv_path=args.csv,
        action=args.action,
        output_dir=args.output_dir
    )