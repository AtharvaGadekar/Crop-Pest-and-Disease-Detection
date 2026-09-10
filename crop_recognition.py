"""Model loading and non-blocking maize pest recognition."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


ARTIFACT_FORMAT_VERSION = 1
ARCHITECTURE = "mobilenet_v3_small"
DEFAULT_CLASS_NAMES = (
    "healthy",
    "fall armyworm",
    "grasshopper",
    "leaf beetle",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_ARTIFACT_KEYS = {
    "format_version",
    "architecture",
    "state_dict",
    "class_names",
    "input_size",
    "mean",
    "std",
    "confidence_threshold",
}


@dataclass(frozen=True)
class PredictionSnapshot:
    status: str
    label: str | None
    confidence: float | None
    confident: bool
    message: str
    updated_at: float


@dataclass
class LoadedClassifier:
    model: nn.Module
    class_names: tuple[str, ...]
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    confidence_threshold: float
    device: torch.device

    def __post_init__(self) -> None:
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def predict_jpeg(self, jpeg: bytes) -> tuple[tuple[float, ...], float]:
        try:
            with Image.open(BytesIO(jpeg)) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ValueError("Could not decode JPEG frame") from error

        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].to("cpu").tolist()
        return tuple(float(value) for value in probabilities), time.monotonic()


def _select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _number_triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Model artifact {name} must contain three numbers")
    try:
        triplet = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Model artifact {name} must contain three numbers") from error
    if not all(math.isfinite(item) for item in triplet):
        raise ValueError(f"Model artifact {name} contains a non-finite value")
    return triplet  # type: ignore[return-value]


def load_artifact(
    path: str | Path,
    device: str | None = None,
    confidence_threshold: float | None = None,
) -> LoadedClassifier:
    artifact_path = Path(path)
    try:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Could not load model artifact: {artifact_path}") from error

    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary")
    missing = REQUIRED_ARTIFACT_KEYS.difference(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing: {', '.join(sorted(missing))}")
    if artifact["format_version"] != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format")
    if artifact["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported model architecture")

    raw_names = artifact["class_names"]
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise ValueError("Model artifact class_names must be a non-empty list")
    class_names = tuple(str(name).strip() for name in raw_names)
    if any(not name for name in class_names) or len(set(class_names)) != len(class_names):
        raise ValueError("Model artifact class names must be unique and non-empty")

    input_size = artifact["input_size"]
    if not isinstance(input_size, int) or input_size < 32 or input_size > 2048:
        raise ValueError("Model artifact input_size is invalid")
    mean = _number_triplet(artifact["mean"], "mean")
    std = _number_triplet(artifact["std"], "std")
    if any(value <= 0 for value in std):
        raise ValueError("Model artifact std values must be positive")

    threshold_value = (
        artifact["confidence_threshold"]
        if confidence_threshold is None
        else confidence_threshold
    )
    try:
        threshold = float(threshold_value)
    except (TypeError, ValueError) as error:
        raise ValueError("Confidence threshold must be a number") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1")

    model = models.mobilenet_v3_small(weights=None)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise ValueError("Unexpected MobileNet classifier structure")
    model.classifier[-1] = nn.Linear(final_layer.in_features, len(class_names))
    try:
        model.load_state_dict(artifact["state_dict"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Model weights do not match artifact metadata") from error

    selected_device = _select_device(device)
    model.to(selected_device)
    model.eval()
    return LoadedClassifier(
        model=model,
        class_names=class_names,
        input_size=input_size,
        mean=mean,
        std=std,
        confidence_threshold=threshold,
        device=selected_device,
    )


class PredictionSmoother:
    def __init__(
        self,
        class_names: Sequence[str],
        confidence_threshold: float,
        window_size: int = 5,
    ) -> None:
        if not class_names:
            raise ValueError("At least one class name is required")
        if window_size < 1:
            raise ValueError("Smoothing window must be at least one")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self.class_names = tuple(class_names)
        self.confidence_threshold = float(confidence_threshold)
        self._history: deque[tuple[float, ...]] = deque(maxlen=window_size)

    def add(
        self,
        probabilities: Sequence[float],
        timestamp: float | None = None,
    ) -> PredictionSnapshot:
        if len(probabilities) != len(self.class_names):
            raise ValueError("Probability count does not match class count")
        vector = tuple(float(value) for value in probabilities)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Probabilities must be finite")
        self._history.append(vector)
        averages = tuple(
            sum(vector[index] for vector in self._history) / len(self._history)
            for index in range(len(self.class_names))
        )
        best_index = max(range(len(averages)), key=averages.__getitem__)
        confidence = averages[best_index]
        label = self.class_names[best_index]
        confident = confidence >= self.confidence_threshold
        message = (
            f"{label.title()} — {confidence:.0%}"
            if confident
            else "Uncertain — move closer"
        )
        return PredictionSnapshot(
            status="ready",
            label=label,
            confidence=confidence,
            confident=confident,
            message=message,
            updated_at=time.monotonic() if timestamp is None else timestamp,
        )


class InferenceWorker:
    def __init__(
        self,
        classifier: LoadedClassifier,
        interval_seconds: float = 0.5,
        stale_after_seconds: float = 3.0,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("Inference interval cannot be negative")
        if stale_after_seconds <= 0:
            raise ValueError("Stale timeout must be positive")
        self.classifier = classifier
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self._smoother = PredictionSmoother(
            classifier.class_names, classifier.confidence_threshold
        )
        self._condition = threading.Condition()
        self._snapshot_lock = threading.Lock()
        self._pending_frame: bytes | None = None
        self._running = True
        self._last_run = 0.0
        self._snapshot = PredictionSnapshot(
            status="waiting",
            label=None,
            confidence=None,
            confident=False,
            message="Waiting for a clear frame",
            updated_at=time.monotonic(),
        )
        self._thread = threading.Thread(
            target=self._run,
            name="crop-recognition",
            daemon=True,
        )
        self._thread.start()

    def submit(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        with self._condition:
            if not self._running:
                return
            self._pending_frame = jpeg
            self._condition.notify()

    def snapshot(self) -> PredictionSnapshot:
        with self._snapshot_lock:
            current = self._snapshot
        if (
            current.status == "ready"
            and time.monotonic() - current.updated_at > self.stale_after_seconds
        ):
            return PredictionSnapshot(
                status="waiting",
                label=None,
                confidence=None,
                confident=False,
                message="Waiting for a clear frame",
                updated_at=current.updated_at,
            )
        return current

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _publish(self, snapshot: PredictionSnapshot) -> None:
        with self._snapshot_lock:
            self._snapshot = snapshot

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending_frame is None:
                    self._condition.wait()
                if not self._running:
                    return
                wait_for = self.interval_seconds - (time.monotonic() - self._last_run)
                if wait_for > 0:
                    self._condition.wait(timeout=wait_for)
                    continue
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue
            try:
                probabilities, timestamp = self.classifier.predict_jpeg(frame)
                self._last_run = time.monotonic()
                self._publish(self._smoother.add(probabilities, timestamp))
            except ValueError:
                self._last_run = time.monotonic()
                continue
            except Exception:
                self._publish(
                    PredictionSnapshot(
                        status="unavailable",
                        label=None,
                        confidence=None,
                        confident=False,
                        message="Recognition unavailable",
                        updated_at=time.monotonic(),
                    )
                )
                with self._condition:
                    self._running = False
                return
