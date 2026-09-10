# Maize Pest Recognition Rover Demo Design

## Goal

Extend the existing Quarky Intellio camera-and-control application with a classroom-ready image classifier. While the rover remains manually driven, the Mac will classify close-up camera frames as one of four maize conditions and display a stable prediction without interrupting rover control.

The four initial classes are:

- healthy
- fall armyworm
- grasshopper
- leaf beetle

This is image-level pest recognition, not pest localization. The interface must not draw bounding boxes or imply that it has located an individual insect.

## Success criteria

The demo is successful when:

1. The existing WASD controls, stop-on-key-release behavior, focus-loss safety stop, and UDP camera stream continue to work.
2. A trained model can be loaded locally on the Mac and classify a complete JPEG frame without network access.
3. Inference runs outside the Pygame event loop so a slow prediction cannot delay motor-control events.
4. The display shows the predicted class, smoothed confidence, and either a confident or uncertain state.
5. The application remains usable when the model is missing, a JPEG is corrupt, or inference fails; rover controls must not depend on the classifier.
6. A camera-free command can evaluate sample images for rehearsal and troubleshooting.

Field-grade generalization, insect counting, bounding boxes, autonomous navigation, and automated treatment are outside this first version.

## Architecture

The existing application remains responsible for UDP frame assembly, rover commands, safety behavior, and Pygame rendering. A separate recognition module owns model loading, preprocessing, inference, temporal smoothing, and prediction state.

The receiver publishes the latest complete JPEG. The UI continues rendering that JPEG at its current rate. At a configurable interval, an inference worker reads the newest frame, discarding older unprocessed frames rather than building a queue. The worker decodes and preprocesses the frame, executes the model, and atomically publishes a prediction snapshot for the UI.

This latest-frame design bounds latency and memory use. Driving and emergency-stop input never wait for image decoding or model execution.

## Components

### Dataset preparation and training

A training command will accept the downloaded Kaggle dataset directory and select folders corresponding to the four maize classes. Dataset contents will not be copied into the project.

The Kaggle mirror spells the grasshopper directory `Maize grasshoper`; the trainer will accept that source spelling while the exported and displayed class name remains `grasshopper`.

The trainer will:

- validate that all required class directories exist and contain readable images;
- create deterministic, stratified training, validation, and test splits;
- apply training-only augmentation for crop, rotation, lighting, and mild blur;
- fine-tune an ImageNet-pretrained MobileNetV3 Small model;
- weight the training loss by class frequency to compensate for the imbalanced source folders;
- use Apple Metal acceleration when available and fall back to CPU;
- retain the checkpoint with the best validation loss;
- report per-class precision, recall, F1, a confusion matrix, and overall test accuracy;
- export one self-describing model artifact containing weights, class order, input size, normalization parameters, and confidence threshold.

The test split is an engineering indicator, not proof of field performance. Before the classroom demonstration, the model must also be tried on frames captured through the Intellio camera.

### Runtime recognizer

The recognizer will expose a small interface that accepts JPEG bytes and returns immutable prediction snapshots. It will not import or control rover hardware.

Predictions will be smoothed across the most recent five successful inferences. The displayed label is the class with the highest mean probability. A result is confident only when its mean probability meets the configured threshold. Otherwise, the UI displays `Uncertain — move closer`.

Inference will be enabled with a command-line model path. Without that option, the controller will retain its current camera-only behavior.

### Controller integration

The Pygame window will add a compact overlay containing:

- `MAIZE PEST RECOGNITION`;
- predicted class;
- confidence percentage;
- a green confident or amber uncertain indicator;
- inference status such as loading, ready, or unavailable.

The worker starts only after the model loads successfully and stops during application cleanup. Model errors are reported visually and on stderr but do not close the controller or change motor commands.

### Offline image command

A separate command will load the same exported artifact and classify one or more JPEG/PNG files. It will print the top prediction and confidence for each file and return a nonzero exit status for invalid input or model-loading failures. This provides a repeatable demonstration check without requiring the rover.

## Data flow

1. Intellio sends JPEG chunks over UDP.
2. `VideoFrameAssembler` publishes a complete JPEG as the receiver's latest frame.
3. Pygame renders the latest JPEG and handles keyboard events as it does today.
4. The inference worker periodically snapshots the same JPEG bytes.
5. The recognizer decodes, resizes, normalizes, and classifies the image.
6. The smoothing component updates the latest prediction snapshot.
7. Pygame reads and draws that snapshot without waiting for inference.

## Error and safety behavior

- Missing or incompatible model: show recognition unavailable and preserve camera and controls.
- Missing dataset class or a class with no readable images: fail the training command with the affected path and a clear message. Report and skip isolated malformed source images.
- Corrupt live JPEG: skip that inference and retain the last valid result temporarily.
- Stale prediction: replace it with a waiting state after a short timeout.
- Worker exception: stop inference, expose the error state, and keep the rover controllable.
- Application exit or focus loss: preserve the existing safe-idle behavior regardless of recognition state.

The classifier will never issue motor commands in this version.

## Manual verification

Automated tests are deferred for this classroom-demo iteration. Manual verification will include:

- training a short smoke-test run on a small subset;
- classifying held-out files with the offline command;
- running the controller without a model to verify backward compatibility;
- running the controller with a model and synthetic/sample frame input where hardware-independent testing permits;
- manually confirming responsive driving and safe stop with the live rover before the classroom demonstration.

## Demo procedure

Use one close-up specimen or clearly identified image card at a time. Stop the rover, fill most of the frame with the subject, hold it steady until the prediction stabilizes, and explain that the result identifies the overall image class rather than locating an insect. An uncertain result is a deliberate safety behavior, not a forced guess.

The dataset source will be credited under CC BY 4.0, following the original Mendeley record even though the Kaggle mirror advertises a more permissive license.
