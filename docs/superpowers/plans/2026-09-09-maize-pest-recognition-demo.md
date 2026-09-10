# Maize Pest Recognition Rover Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four-class maize pest recognition to the existing Quarky Intellio camera-and-control application, with training and offline-classification commands.

**Architecture:** Fine-tune MobileNetV3 Small on four selected Kaggle folders and save weights plus preprocessing metadata in one PyTorch artifact. At runtime, a background latest-frame worker classifies JPEG snapshots and publishes smoothed immutable predictions that the existing Pygame loop can render without delaying rover controls.

**Tech Stack:** Python 3.14, PyTorch, torchvision, Pillow, pygame-ce, Apple MPS with CPU fallback

**Spec:** `docs/superpowers/specs/2026-09-09-maize-pest-recognition-demo-design.md`

## Global Constraints

- Initial labels are `healthy`, `fall armyworm`, `grasshopper`, and `leaf beetle`.
- This is image-level recognition; do not draw boxes or describe results as pest localization.
- The classifier must never issue motor commands.
- Dataset files and trained model artifacts remain outside source control.
- Automated tests are deferred by explicit user instruction; perform the manual verification listed in Task 5.
- The workspace is not a Git repository, so this plan contains no commit steps.

---

## File structure

- `requirements-crop.txt`: runtime and training dependencies.
- `.gitignore`: dataset, checkpoint, report, and local environment exclusions for future repository use.
- `crop_recognition.py`: artifact loading, preprocessing, inference, smoothing, immutable prediction state, and background worker.
- `train_maize_classifier.py`: selected-folder discovery, stratified splitting, MobileNet training, metrics, and artifact export.
- `classify_crop_image.py`: camera-free inference CLI.
- `quarky_wasd.py`: optional recognizer startup, frame submission, overlay rendering, and cleanup.
- `CROP_DEMO.md`: setup, training, controller usage, demo procedure, limitations, and attribution.

### Task 1: Dependencies and artifact contract

**Files:**
- Create: `requirements-crop.txt`
- Create: `.gitignore`
- Create: `crop_recognition.py`

**Interfaces:**
- Produces: `load_artifact(path: str | Path, device: str | None = None) -> LoadedClassifier`
- Produces: `LoadedClassifier.predict_jpeg(jpeg: bytes) -> tuple[tuple[float, ...], float]`
- Produces: `PredictionSnapshot(status: str, label: str | None, confidence: float | None, confident: bool, message: str, updated_at: float)`
- Produces: `PredictionSmoother.add(probabilities: Sequence[float], timestamp: float | None = None) -> PredictionSnapshot`

- [x] **Step 1: Declare dependencies and ignored outputs**

Create `requirements-crop.txt` with `torch`, `torchvision`, `Pillow`, and `pygame-ce`. Add `.venv/`, `path/to/venv/`, `datasets/`, `models/`, `reports/`, `*.pth`, and `*.pt` to `.gitignore` while preserving any existing entries.

- [x] **Step 2: Define and validate the model artifact**

Implement a dictionary artifact with these exact keys:

```python
{
    "format_version": 1,
    "architecture": "mobilenet_v3_small",
    "state_dict": model.state_dict(),
    "class_names": ["healthy", "fall armyworm", "grasshopper", "leaf beetle"],
    "input_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "confidence_threshold": 0.70,
}
```

`load_artifact` must reject missing keys, unsupported format or architecture, duplicate/empty class names, invalid image sizes, mismatched classifier output size, and thresholds outside `[0, 1]`. Select `mps` when available unless an explicit device is supplied, otherwise use CPU.

- [x] **Step 3: Implement JPEG preprocessing and inference**

Decode with Pillow, convert to RGB, resize to 256, center-crop to the artifact input size, convert to tensor, and normalize with artifact values. Run `model.eval()` under `torch.inference_mode()`, apply softmax, and return the probability tuple plus a monotonic timestamp. Raise `ValueError("Could not decode JPEG frame")` for invalid image data.

- [x] **Step 4: Implement five-sample smoothing**

Store at most five probability vectors in a `collections.deque(maxlen=5)`. Average each class across the stored vectors, choose the maximum, compare it with the artifact threshold, and produce either `<Title Case Label> — NN%` or `Uncertain — move closer`. Reject probability vectors whose length differs from the class count.

### Task 2: Background latest-frame inference

**Files:**
- Modify: `crop_recognition.py`

**Interfaces:**
- Consumes: `LoadedClassifier.predict_jpeg`
- Produces: `InferenceWorker(classifier: LoadedClassifier, interval_seconds: float = 0.5, stale_after_seconds: float = 3.0)`
- Produces: `InferenceWorker.submit(jpeg: bytes) -> None`, `snapshot() -> PredictionSnapshot`, and `stop() -> None`

- [x] **Step 1: Implement latest-frame submission**

Protect one `_pending_frame` slot with a `threading.Condition`. `submit` replaces that slot and notifies the worker; it must never append frames to a queue.

- [x] **Step 2: Implement worker lifecycle**

Start one daemon thread in the constructor. Wait for a frame or shutdown, enforce the configured interval using `time.monotonic`, classify the newest frame, smooth the result, and publish the snapshot under a lock. `stop` sets the shutdown flag, notifies the condition, and joins for at most two seconds.

- [x] **Step 3: Implement runtime failure and stale states**

Invalid individual JPEGs retain the last successful prediction. Any other inference exception publishes `status="unavailable"` and `message="Recognition unavailable"`, then stops further inference. `snapshot` returns `status="waiting"` and `message="Waiting for a clear frame"` when the last successful update is more than three seconds old.

### Task 3: Training command

**Files:**
- Create: `train_maize_classifier.py`

**Interfaces:**
- Consumes: four source folders named `Maize healthy`, `Maize fall armyworm`, `Maize grasshoper`, and `Maize leaf beetle`; the dataset's `grasshoper` spelling maps to the displayed label `grasshopper`
- Produces: artifact dictionary defined in Task 1
- Produces CLI: `python train_maize_classifier.py DATASET_DIR --output models/maize_pests.pt`

- [x] **Step 1: Parse and validate training arguments**

Support `dataset_dir`, `--output`, `--epochs` defaulting to 12, `--batch-size` defaulting to 32, `--learning-rate` defaulting to `0.001`, `--seed` defaulting to 42, `--workers` defaulting to 0, `--confidence-threshold` defaulting to `0.70`, and `--limit-per-class` for smoke runs. Report and skip isolated malformed images, and exit clearly when a required folder is missing or contains no readable `.jpg`, `.jpeg`, or `.png` files.

- [x] **Step 2: Build deterministic stratified splits**

Map the four folders to the exact class order in Task 1. Shuffle each class independently with `random.Random(seed)`, reserve 70% for training, 15% for validation, and the remainder for testing, and ensure every split receives at least one image when a class has three or more files.

- [x] **Step 3: Build transforms and loaders**

Use `RandomResizedCrop(224, scale=(0.70, 1.0))`, `RandomHorizontalFlip`, `RandomRotation(12)`, `ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20)`, `ToTensor`, and ImageNet normalization for training. Use resize-to-256, center-crop-224, tensor conversion, and the same normalization for validation and testing.

- [x] **Step 4: Fine-tune MobileNetV3 Small**

Load torchvision's default pretrained weights, replace the final classifier layer with four outputs, optimize all parameters using AdamW and class-frequency-weighted cross-entropy loss, and train on MPS when available. Print epoch training loss, validation loss, and validation accuracy. Save an in-memory CPU copy of the state dict whenever validation loss improves.

- [x] **Step 5: Evaluate and export**

Calculate a 4x4 confusion matrix plus per-class precision, recall, and F1 without adding scikit-learn. Print the matrix and metrics, create the output parent directory, and save the best CPU state dict with the exact metadata contract from Task 1 using `torch.save`.

### Task 4: Offline CLI and live-controller integration

**Files:**
- Create: `classify_crop_image.py`
- Modify: `quarky_wasd.py`

**Interfaces:**
- Consumes: `load_artifact`, `PredictionSmoother`, and `InferenceWorker`
- Produces CLI: `python classify_crop_image.py MODEL IMAGE [IMAGE ...]`
- Extends controller CLI with: `python quarky_wasd.py --model models/maize_pests.pt`

- [x] **Step 1: Implement offline classification**

Load the artifact once, read each image as bytes, call `predict_jpeg`, pass the probabilities through a one-sample smoother, and print `path: label (NN.N%)` or `path: uncertain (NN.N%)`. Continue across invalid images, report each error to stderr, and return status 1 when any input fails.

- [x] **Step 2: Add controller arguments without breaking defaults**

Use `argparse` for optional `--model`, `--inference-interval` defaulting to `0.5`, and `--confidence-threshold`. When no model is given, preserve the existing behavior and dependencies. If loading fails, print the reason and continue with recognition unavailable.

- [x] **Step 3: Submit frames without blocking Pygame**

When a frame exists, call `worker.submit(frame)` before drawing it. Frame submission only swaps a bytes reference under a lock. Stop the worker in the existing `finally` cleanup before closing the receiver.

- [x] **Step 4: Draw the recognition overlay**

Add a semi-transparent panel in the top-left corner with the heading `MAIZE PEST RECOGNITION`, snapshot message, confidence when available, and status colour: green `(70, 190, 90)` for confident, amber `(235, 175, 65)` for uncertain/waiting, and red `(210, 75, 75)` for unavailable. Keep the camera visible behind the panel and do not draw boxes.

### Task 5: Documentation and manual verification

**Files:**
- Create: `CROP_DEMO.md`

**Interfaces:**
- Documents all commands produced by Tasks 1–4.

- [x] **Step 1: Document setup and dataset attribution**

Document creation of `.venv`, installation from `requirements-crop.txt`, Kaggle download command, expected folder names, the original Mendeley DOI `10.17632/bwh3zbpkpv.1`, and CC BY 4.0 attribution. State explicitly that this version classifies complete images and does not locate or count pests.

- [x] **Step 2: Document training and smoke-run commands**

Include a short smoke run using `--epochs 1 --limit-per-class 20`, the full training command, offline classification examples, and how to interpret validation/test metrics without treating them as field accuracy.

- [x] **Step 3: Document live demo operation**

Include commands for camera-only and model-enabled controller modes, close-up framing guidance, the stop-before-classifying procedure, uncertain-state behavior, and the existing motor-safety precautions.

- [x] **Step 4: Perform dependency and syntax verification**

Run:

```bash
python3 -m py_compile quarky_wasd.py crop_recognition.py train_maize_classifier.py classify_crop_image.py
python3 -m pip install -r requirements-crop.txt
python3 -c "import torch, torchvision, PIL, pygame; print(torch.__version__, torch.backends.mps.is_available())"
```

Expected: compilation succeeds, imports succeed, and the final command prints a torch version plus `True` or a documented CPU fallback.

- [x] **Step 5: Perform smoke training and offline inference**

Run the one-epoch, 20-images-per-class training command against the downloaded dataset, then classify at least one held-out image from each of the four classes. Confirm that the artifact loads, all four outputs are possible, and invalid image input exits nonzero without a traceback.

- [ ] **Step 6: Perform live controller compatibility checks (requires connected rover)**

Run the controller without `--model` and confirm unchanged camera/control behavior. Run it with the smoke model, confirm that the overlay moves from waiting to a prediction, verify that driving remains responsive during inference, and verify stop-on-release, focus-loss stop, quit stop, and cleanup. Hardware-dependent checks must be reported as pending if the rover is not connected.
