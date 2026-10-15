"""
Train MobileNetV2 for Kurta Fabric/Material Classification.
Supports train/validation split, metric logging, data augmentation, and GPU acceleration.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets

import config


def get_mobilenet_model(num_classes: int):
    """
    Instantiates MobileNetV2 with pre-trained weights and updates classifier for target classes.
    Gracefully handles different torchvision versions.
    """
    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    except (ImportError, AttributeError):
        model = torchvision.models.mobilenet_v2(pretrained=True)

    # Replace the final linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train(
    dataset_dir: str = str(config.DATASET_DIR),
    output_path: str = str(config.DEFAULT_MODEL_PATH),
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    val_split: float = config.VAL_SPLIT_RATIO,
):
    """Trains the model with train/validation evaluation."""
    device = config.DEVICE
    print(f"[INFO] Using compute device: {device}")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    # Load full dataset with training transforms initially
    full_dataset = datasets.ImageFolder(str(dataset_path), transform=config.TRAIN_TRANSFORMS)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"[INFO] Found {len(full_dataset)} images across {num_classes} classes: {classes}")

    # Check for empty dataset
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in dataset path '{dataset_dir}'.")

    # Train / Validation Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_data, val_data = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_size > 0 else None

    print(f"[INFO] Dataset split: {train_size} training samples, {val_size} validation samples.")

    # Initialize model
    model = get_mobilenet_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    print("\n" + "=" * 50)
    print("Beginning Training Pipeline")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation Phase
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0

        if val_loader and len(val_data) > 0:
            model.eval()
            running_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = running_val_loss / val_total
            epoch_val_acc = (val_correct / val_total) * 100.0
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc

        scheduler.step()

        # Display progress
        val_str = f" | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%" if val_loader else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}%{val_str}")

    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save model state dict (100% compatible with existing scripts)
    torch.save(model.state_dict(), str(output_path_obj))
    print(f"\n[SUCCESS] Model weights saved to: {output_path_obj}")

    # Also save metadata & class mapping for robust inference
    classes_json_path = output_path_obj.parent / "classes.json"
    with open(classes_json_path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_to_idx": full_dataset.class_to_idx}, f, indent=2)
    print(f"[SUCCESS] Class mappings exported to: {classes_json_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kurta Fabric Classification Model")
    parser.add_argument("--dataset", type=str, default=str(config.DATASET_DIR), help="Path to dataset directory")
    parser.add_argument("--output", type=str, default=str(config.DEFAULT_MODEL_PATH), help="Path to save model weights (.pth)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=config.VAL_SPLIT_RATIO, help="Validation split ratio (0.0 to 1.0)")

    args = parser.parse_args()
    train(
        dataset_dir=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
    )