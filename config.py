"""
Configuration and constants for the Kurta Fabric Classification Project.
Provides unified paths, dataset class definitions, transforms, and model hyperparameters.
"""
import os
from pathlib import Path
import torch
from torchvision import transforms

# ==============================================================================
# Project Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "kurta_dataset"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "kurta_model.pth"
DEFAULT_RESULTS_EXCEL = PROJECT_ROOT / "kurta_results.xlsx"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "kurta_results.csv"
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "crawled_images"

# ==============================================================================
# Fabric Classes (Alphabetical order - matches torchvision.datasets.ImageFolder)
# ==============================================================================
CLASSES = ["cotton", "linen", "nylon", "silk"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# ==============================================================================
# Model & Image Hyperparameters
# ==============================================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VAL_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

# ImageNet normalization standards (MobileNetV2 backbone)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Device detection
def get_device():
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# ==============================================================================
# Preprocessing Transforms
# ==============================================================================
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])
