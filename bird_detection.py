"""Local bird detection with bounded, asynchronous JPEG inference.

Coordinates always refer to the source JPEG. No robot commands are sent here.
Heavy dependencies are imported only when detection is enabled.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parent / 'models' / 'yolo11n.pt'
TARGET_NAMES = {'bird', 'pigeon', 'rock pigeon'}


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    label: str
    confidence: float
    direction: str


@dataclass(frozen=True)
class FrameDetections:
    width: int
    height: int
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class DetectionSnapshot:
    status: str = 'loading'
    frame_id: int | None = None
    captured_at: float = 0.0
    jpeg: bytes | None = None
    width: int = 0
    height: int = 0
    detections: tuple[Detection, ...] = ()
    inference_ms: float = 0.0
    error: str | None = None


class LoadedDetector:
    def __init__(self, model, device):
        if model.task != 'detect':
            raise ValueError('Bird model must be an object detection model')
        self.model = model
        self.device = device
        names = model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))
        self.class_ids = [int(key) for key, value in self.names.items()
                          if str(value).strip().lower() in TARGET_NAMES]
        if not self.class_ids:
            raise ValueError('Model has no bird, pigeon, or rock pigeon class')


def load_detector(model_path=DEFAULT_MODEL, device='auto'):
    from ultralytics import YOLO
    from urllib.request import urlopen
    import torch

    path = Path(model_path).expanduser()
    if not path.is_file():
        if path.resolve() != DEFAULT_MODEL:
            raise FileNotFoundError(f'Bird model not found: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download atomically to our exact path. Ultralytics' downloader strips
        # apostrophes, which otherwise changes this project's directory name.
        url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt'
        with tempfile.TemporaryDirectory(dir=path.parent) as download_dir:
            downloaded = Path(download_dir) / path.name
            with urlopen(url, timeout=30) as response, downloaded.open('wb') as output:
                shutil.copyfileobj(response, output)
            downloaded.replace(path)
    if device == 'auto':
        device = ('cuda:0' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')
    if "'" in str(path):
        # Loading also goes through the upstream path sanitizer. A temporary
        # alias preserves the original file and does not change process cwd.
        with tempfile.TemporaryDirectory(prefix='rover-model-') as alias_dir:
            alias = Path(alias_dir) / ('weights' + path.suffix)
            shutil.copyfile(path, alias)
            model = YOLO(str(alias))
    else:
        model = YOLO(str(path))
    return LoadedDetector(model, device)


def calculate_direction(box, frame_width):
    if not math.isfinite(frame_width) or frame_width <= 0:
        raise ValueError('Frame width must be positive and finite')
    center = (box[0] + box[2]) / 2
    if not math.isfinite(center):
        raise ValueError('Box center must be finite')
    if center < frame_width * .4:
        return 'LEFT'
    if center > frame_width * .6:
        return 'RIGHT'
    return 'CENTER'


def detect_birds(jpeg_bytes, detector, confidence_threshold=.5):
    from PIL import Image

    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence must be between 0 and 1')
    try:
        with Image.open(BytesIO(jpeg_bytes), formats=['JPEG']) as source:
            if source.format != 'JPEG':
                raise ValueError('Expected JPEG image')
            image = source.convert('RGB')
    except (OSError, ValueError) as error:
        raise ValueError('Could not decode JPEG frame') from error
    width, height = image.size
    result = detector.model.predict(image, conf=confidence_threshold,
                classes=detector.class_ids, device=detector.device,
                imgsz=640, verbose=False)[0]
    detections = []
    rows = () if result.boxes is None else result.boxes.data.cpu().tolist()
    for x1, y1, x2, y2, confidence, class_id in rows:
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence, class_id)):
            continue
        if int(class_id) not in detector.class_ids or confidence < confidence_threshold:
            continue
        box = (max(0, min(width, x1)), max(0, min(height, y1)),
               max(0, min(width, x2)), max(0, min(height, y2)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(Detection(box, detector.names[int(class_id)], confidence,
                                    calculate_direction(box, width)))
    return FrameDetections(width, height, tuple(detections))


def scale_box(box, source_size, image_rect):
    """Map source xyxy to screen xywh, including letterbox offsets."""
    source_width, source_height = source_size
    left, top, width, height = image_rect
    x1, y1, x2, y2 = box
    return (round(left + x1 * width / source_width),
            round(top + y1 * height / source_height),
            max(1, round((x2 - x1) * width / source_width)),
            max(1, round((y2 - y1) * height / source_height)))


class BirdDetectionWorker:
    def __init__(self, model_path=DEFAULT_MODEL, confidence_threshold=.5,
                 interval_seconds=.2, max_result_age=1.0, detector_factory=load_detector):
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError('Detection interval must be finite and nonnegative')
        if not math.isfinite(max_result_age) or max_result_age <= 0:
            raise ValueError('Maximum result age must be positive and finite')
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        self.interval = interval_seconds
        self.max_age = max_result_age
        self.threshold = confidence_threshold
        self.condition = threading.Condition()
        self.pending = None
        self.last_submitted = -1
        self.running = True
        self.result = DetectionSnapshot()
        self.thread = threading.Thread(target=self._run,
                    args=(model_path, detector_factory), daemon=True, name='bird-detection')
        self.thread.start()

    def submit(self, frame_id, jpeg_bytes, captured_at):
        with self.condition:
            if not self.running or frame_id <= self.last_submitted:
                return
            self.last_submitted = frame_id
            self.pending = (frame_id, jpeg_bytes, captured_at)
            self.condition.notify()

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.condition:
            result = self.result
        if result.status == 'ready' and now - result.captured_at > self.max_age:
            return replace(result, status='stale', jpeg=None, detections=())
        return result

    def _run(self, model_path, detector_factory):
        try:
            detector = detector_factory(model_path)
        except Exception as error:
            with self.condition:
                self.result = DetectionSnapshot(status='unavailable', error=str(error))
                self.pending = None
                self.running = False
            return
        with self.condition:
            self.result = DetectionSnapshot(status='waiting')
        next_run = 0.0
        while True:
            with self.condition:
                while self.running:
                    delay = next_run - time.monotonic()
                    if self.pending is not None and delay <= 0:
                        break
                    self.condition.wait(timeout=max(.001, delay) if self.pending else None)
                if not self.running:
                    return
                frame_id, jpeg, captured_at = self.pending
                self.pending = None
            if time.monotonic() - captured_at > self.max_age:
                continue
            started = time.monotonic()
            try:
                prediction = detect_birds(jpeg, detector, self.threshold)
                result = DetectionSnapshot('ready', frame_id, captured_at, jpeg,
                    prediction.width, prediction.height, prediction.detections,
                    (time.monotonic() - started) * 1000)
            except Exception as error:
                result = DetectionSnapshot(status='error', frame_id=frame_id,
                                           captured_at=captured_at, error=str(error))
            with self.condition:
                if not self.running:
                    return
                self.result = result
            next_run = started + self.interval

    def stop(self):
        with self.condition:
            self.running = False
            self.pending = None
            self.condition.notify_all()
        # A native inference/download cannot be interrupted; the daemon exits
        # after that operation returns. Never hold up the control window on quit.
        self.thread.join(timeout=1.0)
