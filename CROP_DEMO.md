# Quarky Maize Pest Recognition Demo

This project classifies an entire Quarky Intellio camera frame as one of four maize conditions:

- healthy
- fall armyworm
- grasshopper
- leaf beetle

It does not locate pests, draw bounding boxes, or count insects. Keep one leaf or demonstration image large and clear in the camera frame.

## 1. Set up Python

From this project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-crop.txt
```

PyTorch automatically uses Apple's MPS acceleration when it is available. It otherwise falls back to CPU.

## 2. Download the dataset

Install and configure the Kaggle command-line tool if it is not already available, then download outside the source files:

```bash
mkdir -p datasets/crop-pest-disease
kaggle datasets download \
  nirmalsankalana/crop-pest-and-disease-detection \
  --unzip \
  --path datasets/crop-pest-disease
```

The trainer expects these exact directories below the dataset path:

```text
Maize healthy/
Maize fall armyworm/
Maize grasshoper/
Maize leaf beetle/
```

`grasshoper` is the spelling used by the downloaded dataset. The application presents the corrected label `grasshopper`.

The original dataset is **Dataset for Crop Pest and Disease Detection**, DOI [10.17632/bwh3zbpkpv.1](https://doi.org/10.17632/bwh3zbpkpv.1), by Patrick Mensah Kwabena and collaborators. Credit that source under CC BY 4.0 in classroom materials. The original dataset contains field images collected in Ghana; validation metrics should not be presented as proof of performance on local crops or general field conditions.

## 3. Train the model

First verify the pipeline using 20 images from each class and one epoch:

```bash
.venv/bin/python train_maize_classifier.py \
  datasets/crop-pest-disease \
  --epochs 1 \
  --limit-per-class 20 \
  --output models/maize_pests_smoke.pt
```

Then train the full classroom model:

```bash
.venv/bin/python train_maize_classifier.py \
  datasets/crop-pest-disease \
  --epochs 12 \
  --output models/maize_pests.pt
```

Training prints validation accuracy after each epoch. At the end it prints a test confusion matrix plus precision, recall, and F1 for each class. These measurements test held-out dataset images only. Always try the exported model on frames from the actual Intellio camera before the demonstration.

The trainer automatically weights the loss by class frequency so the larger leaf-beetle folder does not dominate the smaller healthy and fall-armyworm classes.

The source archive contains a small number of malformed JPEGs. The trainer reports and skips those files; it stops only if a required class has no readable images.

## 4. Check images without the rover

```bash
.venv/bin/python classify_crop_image.py \
  models/maize_pests.pt \
  "datasets/crop-pest-disease/Maize healthy/healthy1_.jpg"
```

Pass multiple image paths to classify them in one invocation. The exact filenames vary; choose any JPEG or PNG from the relevant class folders.

## 5. Run the rover

Camera and controls only:

```bash
.venv/bin/python quarky_wasd.py
```

Camera, controls, and recognition overlay:

```bash
.venv/bin/python quarky_wasd.py --model models/maize_pests.pt
```

If the classroom lighting differs from the dataset, lower or raise the confidence threshold after rehearsal:

```bash
.venv/bin/python quarky_wasd.py \
  --model models/maize_pests.pt \
  --confidence-threshold 0.65
```

## Demo procedure

1. Keep the rover lifted off the ground for the first control check.
2. Confirm that releasing W or S stops the motors and losing window focus triggers safe idle.
3. Drive to the specimen or image card and stop the rover.
4. Place one subject against a simple background and fill roughly 60–80% of the frame.
5. Hold the view steady while five predictions accumulate.
6. Read the displayed class and confidence. If the display says `Uncertain — move closer`, improve framing instead of forcing a result.
7. Describe the result as image-level pest recognition, not insect localization or a definitive agricultural diagnosis.

The recognition module never sends motor commands. A missing or invalid model disables recognition while leaving the existing camera and manual controls available.
