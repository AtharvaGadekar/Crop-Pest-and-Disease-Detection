#!/usr/bin/env python3
"""Train the full crop pest and disease classifier."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from crop_recognition import (
    ARCHITECTURE,
    ARTIFACT_FORMAT_VERSION,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


CLASS_FOLDERS = {
    "cashew anthracnose": "Cashew anthracnose",
    "cashew gummosis": "Cashew gumosis",
    "cashew healthy": "Cashew healthy",
    "cashew leaf miner": "Cashew leaf miner",
    "cashew red rust": "Cashew red rust",
    "cassava bacterial blight": "Cassava bacterial blight",
    "cassava brown spot": "Cassava brown spot",
    "cassava green mite": "Cassava green mite",
    "cassava healthy": "Cassava healthy",
    "cassava mosaic": "Cassava mosaic",
    "maize fall armyworm": "Maize fall armyworm",
    "maize grasshopper": "Maize grasshoper",
    "maize healthy": "Maize healthy",
    "maize leaf beetle": "Maize leaf beetle",
    "maize leaf blight": "Maize leaf blight",
    "maize leaf spot": "Maize leaf spot",
    "maize streak virus": "Maize streak virus",
    "tomato healthy": "Tomato healthy",
    "tomato leaf blight": "Tomato leaf blight",
    "tomato leaf curl": "Tomato leaf curl",
    "tomato septoria leaf spot": "Tomato septoria leaf spot",
    "tomato verticillium wilt": "Tomato verticulium wilt",
}
CLASS_NAMES = tuple(CLASS_FOLDERS)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    label: int


class PestImageDataset(Dataset):
    def __init__(self, items: Sequence[ImageItem], transform) -> None:
        self.items = tuple(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(item.path) as source:
            image = source.convert("RGB")
        return self.transform(image), item.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3 Small on all 22 crop dataset classes."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("models/crop_all_classes.pt")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/crop_all_classes")
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1")
    if args.limit_per_class is not None and args.limit_per_class < 3:
        parser.error("--limit-per-class must be at least 3")
    return args


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_images(dataset_dir: Path) -> dict[str, list[Path]]:
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    discovered: dict[str, list[Path]] = {}
    for class_name in CLASS_NAMES:
        folder = dataset_dir / CLASS_FOLDERS[class_name]
        if not folder.is_dir():
            raise ValueError(f"Required class directory is missing: {folder}")
        candidate_paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not candidate_paths:
            raise ValueError(f"Class directory contains no images: {folder}")
        paths: list[Path] = []
        for path in candidate_paths:
            try:
                with Image.open(path) as image:
                    # Decode pixels now: Image.verify() can accept JPEG headers
                    # whose compressed image stream fails later in __getitem__.
                    image.convert("RGB").load()
            except (OSError, UnidentifiedImageError, ValueError):
                print(f"Skipping unreadable training image: {path}", file=sys.stderr)
                continue
            paths.append(path)
        if not paths:
            raise ValueError(f"Class directory has no readable images: {folder}")
        discovered[class_name] = paths
    return discovered


def build_splits(
    images: dict[str, list[Path]],
    seed: int,
    limit_per_class: int | None,
) -> tuple[list[ImageItem], list[ImageItem], list[ImageItem]]:
    train: list[ImageItem] = []
    validation: list[ImageItem] = []
    test: list[ImageItem] = []
    for label, class_name in enumerate(CLASS_NAMES):
        paths = list(images[class_name])
        random.Random(f"{seed}:{class_name}").shuffle(paths)
        if limit_per_class is not None:
            paths = paths[:limit_per_class]
        if len(paths) < 3:
            raise ValueError(f"Class {class_name!r} needs at least three images")

        train_count = max(1, int(len(paths) * 0.70))
        validation_count = max(1, int(len(paths) * 0.15))
        if train_count + validation_count >= len(paths):
            train_count = len(paths) - 2
            validation_count = 1
        train.extend(ImageItem(path, label) for path in paths[:train_count])
        validation.extend(
            ImageItem(path, label)
            for path in paths[train_count : train_count + validation_count]
        )
        test.extend(
            ImageItem(path, label) for path in paths[train_count + validation_count :]
        )

    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    random.Random(seed + 2).shuffle(test)
    return train, validation, test


def build_transforms():
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def build_model() -> nn.Module:
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise RuntimeError("Unexpected MobileNet classifier structure")
    model.classifier[-1] = nn.Linear(final_layer.in_features, len(CLASS_NAMES))
    return model


def balanced_class_weights(items: Sequence[ImageItem]) -> torch.Tensor:
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    for item in items:
        counts[item.label] += 1
    if torch.any(counts == 0):
        raise ValueError("Every class must have at least one training image")
    return counts.sum() / (len(CLASS_NAMES) * counts)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += batch_count
    return total_loss / count, correct / count


def evaluate_confusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    matrix = torch.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64
    )
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).to("cpu")
            for actual, predicted in zip(labels, predictions):
                matrix[int(actual), int(predicted)] += 1
    return matrix


def print_metrics(matrix: torch.Tensor) -> None:
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(matrix)
    print("\nPer-class metrics:")
    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum()) - true_positive
        false_negative = int(matrix[index, :].sum()) - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        print(
            f"  {class_name:16s} precision={precision:.3f} "
            f"recall={recall:.3f} f1={f1:.3f}"
        )
    accuracy = int(matrix.diag().sum()) / max(1, int(matrix.sum()))
    print(f"Overall test accuracy: {accuracy:.3f}")


def save_graphs(
    history: dict[str, list[float]],
    matrix: torch.Tensor,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_axis.plot(epochs, history["train_loss"], marker="o", label="Training")
    loss_axis.plot(epochs, history["validation_loss"], marker="o", label="Validation")
    loss_axis.set_title("Loss by epoch")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    accuracy_axis.plot(
        epochs, history["train_accuracy"], marker="o", label="Training"
    )
    accuracy_axis.plot(
        epochs, history["validation_accuracy"], marker="o", label="Validation"
    )
    accuracy_axis.set_title("Accuracy by epoch")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend()
    figure.tight_layout()
    history_path = report_dir / "training_history.png"
    figure.savefig(history_path, dpi=180)
    plt.close(figure)

    normalized = matrix.to(torch.float32)
    normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(1)
    figure, axis = plt.subplots(figsize=(15, 13))
    image = axis.imshow(normalized.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_title("Normalized test confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    positions = range(len(CLASS_NAMES))
    axis.set_xticks(positions, CLASS_NAMES, rotation=90, fontsize=7)
    axis.set_yticks(positions, CLASS_NAMES, fontsize=7)
    figure.colorbar(image, ax=axis, label="Fraction of actual class")
    figure.tight_layout()
    confusion_path = report_dir / "confusion_matrix.png"
    figure.savefig(confusion_path, dpi=180)
    plt.close(figure)

    print(f"Saved training graph: {history_path}")
    print(f"Saved confusion graph: {confusion_path}")


def main() -> int:
    args = parse_args()
    try:
        images = discover_images(args.dataset_dir)
        train_items, validation_items, test_items = build_splits(
            images, args.seed, args.limit_per_class
        )
    except ValueError as error:
        print(f"Training setup error: {error}", file=sys.stderr)
        return 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_transform, evaluation_transform = build_transforms()
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(
        PestImageDataset(train_items, train_transform), shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        PestImageDataset(validation_items, evaluation_transform),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        PestImageDataset(test_items, evaluation_transform),
        shuffle=False,
        **loader_options,
    )

    device = select_device()
    print(f"Device: {device}")
    print(
        f"Images: train={len(train_items)} validation={len(validation_items)} "
        f"test={len(test_items)}"
    )
    model = build_model().to(device)
    class_weights = balanced_class_weights(train_items).to(device)
    print(
        "Class weights: "
        + ", ".join(
            f"{name}={weight:.3f}"
            for name, weight in zip(CLASS_NAMES, class_weights.to("cpu"))
        )
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.3f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().to("cpu").clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        print("Training did not produce a checkpoint", file=sys.stderr)
        return 3
    model.load_state_dict(best_state)
    model.to(device)
    matrix = evaluate_confusion(model, test_loader, device)
    print_metrics(matrix)
    save_graphs(history, matrix, args.report_dir)

    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "state_dict": copy.deepcopy(best_state),
        "class_names": list(CLASS_NAMES),
        "input_size": 224,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "confidence_threshold": args.confidence_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"Saved model artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
