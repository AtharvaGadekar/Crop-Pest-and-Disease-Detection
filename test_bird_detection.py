"""Behavior tests; only the expensive neural-network prediction is substituted."""
import threading
import time
import unittest
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

import bird_detection as birds


def jpeg_image():
    stream = BytesIO()
    Image.new('RGB', (100, 80), 'white').save(stream, format='JPEG')
    return stream.getvalue()


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class FakeModel:
    names = {0: 'person', 14: 'bird', 80: 'Rock Pigeon'}
    task = 'detect'

    def __init__(self, rows=()):
        self.rows = rows
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.images = []

    def predict(self, image, **kwargs):
        self.images.append(image.size)
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError('Test inference timed out')
        return [SimpleNamespace(boxes=SimpleNamespace(data=Rows(self.rows)))]


class BirdDetectionTests(unittest.TestCase):
    def test_default_download_preserves_apostrophe_path(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Maker's Lab" / 'models' / 'yolo11n.pt'
            def read_model(filename):
                self.assertEqual(Path(filename).read_bytes(), b'model contents')
                self.assertNotIn("'", str(filename))
                return FakeModel()
            with patch.object(birds, 'DEFAULT_MODEL', path.resolve()), \
                 patch('urllib.request.urlopen', return_value=BytesIO(b'model contents')), \
                 patch('ultralytics.YOLO', side_effect=read_model):
                birds.load_detector(path, device='cpu')
            self.assertEqual(path.read_bytes(), b'model contents')

    def test_direction_boundaries(self):
        for cx, expected in [(0, 'LEFT'), (39, 'LEFT'), (40, 'CENTER'),
                             (60, 'CENTER'), (61, 'RIGHT'), (100, 'RIGHT')]:
            with self.subTest(cx=cx):
                self.assertEqual(birds.calculate_direction((cx, 0, cx, 10), 100), expected)
        with self.assertRaises(ValueError):
            birds.calculate_direction((0, 0, 1, 1), 0)

    def test_filters_classes_confidence_and_invalid_boxes(self):
        model = FakeModel([
            [0, 10, 20, 30, .9, 14], [70, 10, 90, 30, .8, 80],
            [40, 10, 60, 30, .99, 0], [40, 10, 60, 30, .2, 14],
            [50, 10, 20, 30, .9, 14], [0, 0, float('nan'), 30, .9, 14],
        ])
        result = birds.detect_birds(jpeg_image(), birds.LoadedDetector(model, 'cpu'), .5)
        self.assertEqual((result.width, result.height), (100, 80))
        self.assertEqual([(d.label, d.direction) for d in result.detections],
                         [('bird', 'LEFT'), ('Rock Pigeon', 'RIGHT')])
        self.assertEqual(result.detections[0].box, (0, 10, 20, 30))

    def test_empty_predictions_and_invalid_jpeg(self):
        detector = birds.LoadedDetector(FakeModel(), 'cpu')
        self.assertEqual(birds.detect_birds(jpeg_image(), detector, .5).detections, ())
        with self.assertRaisesRegex(ValueError, 'JPEG'):
            birds.detect_birds(b'not an image', detector, .5)

    def test_rejects_model_without_supported_classes(self):
        model = FakeModel()
        model.names = {0: 'person'}
        with self.assertRaisesRegex(ValueError, 'bird|pigeon'):
            birds.LoadedDetector(model, 'cpu')

    def test_box_transform_accounts_for_letterboxing(self):
        self.assertEqual(birds.scale_box((10, 20, 30, 40), (100, 80),
                                        (0, 100, 500, 400)), (50, 200, 100, 100))

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.005)
        self.fail('Worker did not produce the expected result')

    def test_worker_drops_pending_frames_and_deduplicates_submissions(self):
        model = FakeModel([[0, 10, 20, 30, .9, 14]])
        model.release.clear()
        worker = birds.BirdDetectionWorker('unused', interval_seconds=0,
                    detector_factory=lambda _: birds.LoadedDetector(model, 'cpu'))
        try:
            jpeg = jpeg_image()
            worker.submit(1, jpeg, time.monotonic())
            self.assertTrue(model.entered.wait(2))
            worker.submit(2, jpeg, time.monotonic())
            worker.submit(3, jpeg, time.monotonic())
            worker.submit(3, jpeg, time.monotonic())
            model.release.set()
            snapshot = self.wait_for(lambda: (s if (s := worker.snapshot()).frame_id == 3 else None))
            self.assertEqual(snapshot.jpeg, jpeg)
            self.assertEqual(len(model.images), 2)
            self.assertEqual(snapshot.detections[0].direction, 'LEFT')
        finally:
            model.release.set()
            worker.stop()
        self.assertFalse(worker.thread.is_alive())

    def test_stale_result_clears_boxes_and_source_image(self):
        worker = birds.BirdDetectionWorker('unused', interval_seconds=0,
                    detector_factory=lambda _: birds.LoadedDetector(FakeModel(), 'cpu'))
        try:
            worker.submit(1, jpeg_image(), time.monotonic())
            ready = self.wait_for(lambda: (s if (s := worker.snapshot()).status == 'ready' else None))
            stale = worker.snapshot(now=ready.captured_at + 2)
            self.assertEqual(stale.status, 'stale')
            self.assertEqual(stale.detections, ())
            self.assertIsNone(stale.jpeg)
        finally:
            worker.stop()

    def test_model_failure_is_visible_without_crashing_caller(self):
        def fail(_):
            raise RuntimeError('Missing model')
        worker = birds.BirdDetectionWorker('unused', detector_factory=fail)
        try:
            result = self.wait_for(lambda: (s if (s := worker.snapshot()).status == 'unavailable' else None))
            self.assertIn('Missing model', result.error)
            self.assertEqual(result.detections, ())
        finally:
            worker.stop()

    def test_bad_frame_reports_error_and_next_frame_recovers(self):
        worker = birds.BirdDetectionWorker('unused', interval_seconds=0,
                    detector_factory=lambda _: birds.LoadedDetector(FakeModel(), 'cpu'))
        try:
            worker.submit(1, b'invalid', time.monotonic())
            self.wait_for(lambda: worker.snapshot().status == 'error')
            worker.submit(2, jpeg_image(), time.monotonic())
            result = self.wait_for(lambda: (s if (s := worker.snapshot()).status == 'ready' else None))
            self.assertEqual(result.frame_id, 2)
            self.assertEqual(result.detections, ())
        finally:
            worker.stop()


if __name__ == '__main__':
    unittest.main()
