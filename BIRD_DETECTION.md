# Bird detection on the rover camera

Run from this project directory:

```bash
.venv/bin/python -m pip install -r requirements-bird.txt
.venv/bin/python quarky_wasd.py --detect-birds
```

Connect to the rover's network and start the existing PictoBlox camera forwarding
first. The receiver listens on `127.0.0.1:6000`, using the existing `SIZE:<bytes>`
header followed by JPEG chunks. Only one controller should listen on that port.
The rover IP and steering settings remain in `quarky_wasd.py`.

The default YOLO11n model is downloaded from the official Ultralytics release into
`models/yolo11n.pt` on first use. The model has already been cached on this workspace.
Detection runs locally on the computer; camera frames are not uploaded. Device
selection prefers CUDA, then Apple MPS, then CPU. Initial loading and the first
inference can take several seconds; the camera and controls remain available.

The default model recognizes **bird**, not pigeon species. A detected pigeon is
therefore labeled **BIRD**. This mode displays positions and does not steer toward
birds. W/S drive, A/D steer, releasing keys stops/centers as before, Space stops,
and Q quits. Keep the camera window focused for keyboard control.

## Display

Each bird gets a box, confidence, and LEFT / CENTER / RIGHT label:

- LEFT: box center is below 40% of the source image width.
- CENTER: box center is between 40% and 60%, inclusive.
- RIGHT: box center is above 60%.

Every qualifying bird is labeled individually. Direction is relative to the
camera image; verify the physical orientation with the actual camera feed.

To keep boxes aligned, the view shows the most recently analyzed JPEG with its
results. The footer reports its age since this computer received it and inference
time. This is a delayed detection view, usually updating up to five times per
second by default; it does not measure latency inside the rover or PictoBlox.
New incoming frames replace pending work instead of accumulating in a queue.

Results older than one second are hidden and the latest camera frame is shown
while waiting for a fresh detection. With no new frames for two seconds, the
image and boxes clear and **STREAM LOST** appears. Fresh video resumes detection
automatically. **NO BIRD DETECTED** means inference completed on a fresh frame
without a qualifying detection. Model/decode errors appear in the footer, with
the full error in the terminal. Restart after correcting a model-loading error.

## Options

```bash
# More sensitive: may produce more false detections.
.venv/bin/python quarky_wasd.py --detect-birds --bird-confidence 0.35

# Attempt inference more frequently if the computer can keep up.
.venv/bin/python quarky_wasd.py --detect-birds --bird-interval 0.1

# A future trained Ultralytics YOLO .pt detection model.
.venv/bin/python quarky_wasd.py --detect-birds --bird-model models/pigeon.pt
```

Custom models must expose at least one class named `bird`, `pigeon`, or
`rock pigeon` (case insensitive). Their actual class name is displayed. Arbitrary
missing model paths report an error rather than downloading a different model.
`--bird-model` requires `--detect-birds`. Default confidence is 0.5 and the minimum
interval between inference starts is 0.2 seconds. Slow inference determines the
actual update rate.

The existing crop `--model` argument retains its meaning. Both can be enabled,
with the crop panel at the top and bird status at the bottom. Running both uses
more compute. Starting without `--detect-birds` keeps bird detection disabled.

## Installation notes

The requirements use the official `ultralytics-opencv-headless` distribution,
which provides the same `ultralytics` Python API without OpenCV window libraries.
Pygame supplies the display. OpenCV headless is pinned to 4.11.0.86: the tested
5.0.0.93 macOS wheel still bundled SDL despite its headless name.
Do not install both OpenCV GUI and headless packages
in the same environment: they supply the same `cv2` module. If you previously
installed the standard Ultralytics package for this feature, replace it:

```bash
.venv/bin/python -m pip uninstall ultralytics opencv-python
.venv/bin/python -m pip install --force-reinstall --no-deps ultralytics-opencv-headless==8.4.147 opencv-python-headless==4.11.0.86
.venv/bin/python -m pip install -r requirements-bird.txt
```

The older `obj.py` uses OpenCV windows, so use a separate environment with GUI
OpenCV if you run that script. The current `quarky_wasd.py` uses Pygame.

## Verification

```bash
.venv/bin/python -m unittest -v
.venv/bin/python -m pip check
```

Tests cover direction boundaries, filtering, multiple birds, invalid JPEGs,
model errors, worker recovery, duplicate/pending frames, stale results,
local UDP assembly and timestamps, resized box coordinates, argument validation,
and existing keyboard behavior. Neural-network calls are substituted in unit
tests; real model inference is checked separately.

Before relying on the live view, check a bird at left, center, and right, empty
scenes, two birds, near/far distances, window resizing, and stopping/restarting
PictoBlox forwarding. Confirm physical left/right and responsiveness of key
release and Space. These hardware checks require the connected rover; a sample
photograph cannot establish field accuracy.

## Code map

- `bird_detection.py`: model loading, JPEG detection, direction/coordinate math,
  immutable results, and the background worker.
- `quarky_wasd.py`: timestamped camera frames, command-line options, source-frame
  selection, and overlays in the existing manual-control loop.
- `test_bird_detection.py`, `test_quarky_wasd.py`: behavior and integration tests.

References: [Ultralytics installation](https://docs.ultralytics.com/quickstart/),
[prediction API](https://docs.ultralytics.com/modes/predict/), and
[COCO classes](https://docs.ultralytics.com/datasets/detect/coco/).

Implementation verification on this Mac: all 20 tests passed, dependency checks
passed, and a simulated local UDP JPEG stream reached real YOLO inference and
the Pygame display. Recorded control commands verified W/key release and quit
cleanup; no commands were sent to the physical rover during these tests.
A pigeon photograph was detected as `bird` at approximately 91% confidence.
This is a smoke test, not a measured accuracy result on rover footage.

The local preview is `reports/bird-detection/preview.png`. Sample photo:
[Rock dove — natures pics.jpg](https://commons.wikimedia.org/wiki/File:Rock_dove_-_natures_pics.jpg)
by Alan D. Wilson, [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/).
The preview resizes it and overlays detection results. Test media and model
weights are excluded from Git.
