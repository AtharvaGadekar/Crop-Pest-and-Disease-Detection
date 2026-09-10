import cv2
import numpy as np
import socket
import threading
import time
import os
import math

# ================= CONFIGURATION =================
QUARKY_IP = "192.168.137.199"
STREAM_PORT = 6000
LOCALHOST = "127.0.0.1"
CONTROL_PORT = 5006

MAX_UDP_PACKET = 65536

# Motor Parameters
MAX_SPEED = 50
MIN_SPEED = 0
diff = 5

# Contour filtering (obstacle avoidance)
MIN_CONTOUR_AREA = 150

SERVO_CENTER = 90
# Dynamic servo range — how far left/right we can steer
SERVO_LEFT_MAX = 55    # hardest left
SERVO_RIGHT_MAX = 125  # hardest right

FORWARD_SPEED = 60

# How long (seconds) to keep turning AFTER the obstacle leaves frame
OBSTACLE_PERSISTENCE_SEC = 0.8

last_servo_angle = None

# ================= TRAFFIC SIGN CONFIGURATION =================
# Model file must sit right next to this script (Tobi.py). It's the
# pre-trained HOG+SVM model from hoanglehaithanh/Traffic-Sign-Detection.
SVM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_svm.dat")

SIGN_SIZE = 32              # training image size used by the model (do not change)
SIGN_MIN_SIZE_COMPONENTS = 300     # min connected-component size to keep as a candidate
SIGN_SIMILARITY_THRESHOLD = 0.65   # how "circular" a contour must be to be a sign candidate
SIGN_DISTANCE_THRESHOLD = 15       # ignore candidates smaller than this (pixels)

SIGN_CONFIRM_FRAMES = 3     # require this many consecutive matching frames before acting
SIGN_ACTION_HOLD_SEC = 2.5  # how long a triggered action (e.g. STOP) stays in force
SIGN_COOLDOWN_SEC = 4.0     # ignore repeat triggers of the same sign for this long after acting

# What each class means, based on inspecting the actual dataset/ folders in the repo.
# Classes 1-9, 11, 12 were confirmed by looking at sample images. Class 10 only had
# a handful of ambiguous samples — verify it yourself (dataset/10/) before trusting it.
#
# action:
#   STOP        -> halt the robot for SIGN_ACTION_HOLD_SEC
#   BIAS_LEFT   -> steer hard left while the action is active
#   BIAS_RIGHT  -> steer hard right while the action is active
#   SET_SPEED   -> change forward speed to "speed" while the action is active
#   IGNORE      -> detected but no robot behaviour attached (log only)
SIGN_INFO = {
    1:  {"name": "STOP",                "action": "STOP"},
    2:  {"name": "TURN LEFT AHEAD",     "action": "BIAS_LEFT"},
    3:  {"name": "TURN RIGHT AHEAD",    "action": "BIAS_RIGHT"},
    4:  {"name": "NO LEFT TURN",        "action": "BIAS_RIGHT"},
    5:  {"name": "NO RIGHT TURN",       "action": "BIAS_LEFT"},
    6:  {"name": "NO ENTRY",            "action": "STOP"},
    7:  {"name": "SPEED LIMIT 40",      "action": "SET_SPEED", "speed": 40},
    8:  {"name": "MIN SPEED 30",        "action": "SET_SPEED", "speed": 45},
    9:  {"name": "KEEP RIGHT",          "action": "BIAS_RIGHT"},
    10: {"name": "UNKNOWN SIGN (verify dataset/10)", "action": "IGNORE"},
    11: {"name": "END OF MIN SPEED 30", "action": "SET_SPEED", "speed": FORWARD_SPEED},
    12: {"name": "KEEP LEFT",           "action": "BIAS_LEFT"},
}
# class 0 = background / not-a-sign, always ignored

# ================= UDP NETWORK HANDLERS =================

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_udp_command(command_str):
    try:
        udp_sock.sendto(command_str.encode(), (QUARKY_IP, CONTROL_PORT))
    except Exception as e:
        print(f"UDP Send Error: {e}")

def initialize_board():
    print("Initializing motor expansion board via UDP...")
    send_udp_command("frame/2/201/1")
    time.sleep(0.5)
    print("Initialization command sent.")

def control_motors(left_speed, right_speed):
    m1_speed = int(max(MIN_SPEED, min(MAX_SPEED, left_speed)))
    m2_speed = int(max(MIN_SPEED, min(MAX_SPEED, right_speed))) + diff
    cmd = f"frame/2/201/10/1/1/{m1_speed}/{m2_speed}"
    send_udp_command(cmd)

def set_servo_angle(angle):
    global last_servo_angle
    angle = int(max(0, min(180, angle)))
    if angle == last_servo_angle:
        return
    last_servo_angle = angle
    cmd = f"frame/2/33/0/{angle}"
    send_udp_command(cmd)

# ================= VIDEO RECEIVER =================

class VideoReceiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((LOCALHOST, port))
        self.sock.settimeout(1.0)
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.packets_received = 0
        self.frames_decoded = 0
        self.last_debug_print = time.time()
        self.thread = threading.Thread(target=self.receive_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"UDP Video Receiver listening on {LOCALHOST}:{port}")

    def receive_loop(self):
        current_buffer = bytearray()
        expected_size = 0
        receiving = False
        addr = ("?", 0)

        while self.running:
            try:
                data, addr = self.sock.recvfrom(MAX_UDP_PACKET)
                self.packets_received += 1

                if data.startswith(b"SIZE:"):
                    try:
                        header = data.decode().strip()
                        expected_size = int(header.split(":")[1])
                        current_buffer = bytearray()
                        receiving = True
                    except Exception:
                        receiving = False
                elif receiving:
                    current_buffer.extend(data)
                    if len(current_buffer) >= expected_size:
                        np_arr = np.frombuffer(current_buffer, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.latest_frame = frame
                            self.frames_decoded += 1
                        receiving = False
                        current_buffer = bytearray()

                now = time.time()
                if now - self.last_debug_print > 2.0:
                    print(f"[VideoReceiver] packets={self.packets_received} "
                          f"frames={self.frames_decoded} addr={addr}")
                    self.last_debug_print = now

            except socket.timeout:
                print("[VideoReceiver] No UDP packet in last 1s.")
                continue
            except Exception as e:
                print(f"[VideoReceiver] Error: {e}")

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self.sock.close()

# ============================================
# COLOUR MASKS (obstacle avoidance — unchanged)
# ============================================

def get_red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 80, 80]),   np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(red1, red2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_blue_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 60, 60]), np.array([140, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ============================================
# DYNAMIC SERVO ANGLE CALCULATOR (obstacle avoidance — unchanged)
# ============================================

def calc_servo_angle(cx, frame_width, direction):
    norm = cx / frame_width  # 0.0 = far left, 1.0 = far right
    CURVE = 2.0

    if direction == "left":
        t = norm ** (1 / CURVE)
        angle = SERVO_CENTER - t * (SERVO_CENTER - SERVO_LEFT_MAX)
    else:
        t = (1.0 - norm) ** (1 / CURVE)
        angle = SERVO_CENTER + t * (SERVO_RIGHT_MAX - SERVO_CENTER)

    return int(np.clip(angle, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

# ============================================
# TRAFFIC SIGN DETECTION + CLASSIFICATION
# Ported from hoanglehaithanh/Traffic-Sign-Detection (main.py / classification.py)
# ============================================

def sign_contrast_limit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels = list(channels)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def sign_laplacian_of_gaussian(image):
    log_image = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(log_image, cv2.COLOR_BGR2GRAY)
    log_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)
    log_image = cv2.convertScaleAbs(log_image)
    return log_image

def sign_binarization(image):
    return cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]

def sign_preprocess(image):
    image = sign_contrast_limit(image)
    image = sign_laplacian_of_gaussian(image)
    image = sign_binarization(image)
    return image

def sign_remove_small_components(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img2 = np.zeros(output.shape, dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def sign_remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_blue  = cv2.inRange(hsv, np.array([100, 128, 0]),   np.array([215, 255, 255]))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 128], dtype=np.uint8),
                                   np.array([255, 255, 255], dtype=np.uint8))
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8),
                                   np.array([170, 150, 50], dtype=np.uint8))

    mask = cv2.bitwise_or(cv2.bitwise_or(mask_blue, mask_white), mask_black)
    return mask

def sign_find_contours(image):
    cnts, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts

def sign_contour_is_sign(perimeter, centroid, threshold):
    result = []
    for p in perimeter:
        p = p[0]
        distance = math.sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
        result.append(distance)
    max_value = max(result)
    if max_value == 0:
        return False, 0
    signature = [float(dist) / max_value for dist in result]
    temp = sum((1 - s) for s in signature) / len(signature)
    if temp < threshold:
        return True, max_value + 2
    return False, max_value + 2

def sign_crop(image, coordinate):
    height, width = image.shape[:2]
    top = max(int(coordinate[0][1]), 0)
    bottom = min(int(coordinate[1][1]), height - 1)
    left = max(int(coordinate[0][0]), 0)
    right = min(int(coordinate[1][0]), width - 1)
    return image[top:bottom, left:right]

def sign_find_largest(image, contours, threshold, distance_threshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = sign_contour_is_sign(c, [cX, cY], 1 - threshold)
        if is_sign and distance > max_distance and distance > distance_threshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
            sign = sign_crop(image, coordinate)
    return sign, coordinate

def sign_deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIGN_SIZE * skew], [0, 1, 0]])
    return cv2.warpAffine(img, M, (SIGN_SIZE, SIGN_SIZE),
                           flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

def sign_get_hog():
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                              derivAperture, winSigma, histogramNormType,
                              L2HysThreshold, gammaCorrection, nlevels, signedGradient)

def sign_classify(model, hog, bgr_crop):
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (SIGN_SIZE, SIGN_SIZE))
    img = sign_deskew(img)
    descriptor = hog.compute(img)
    descriptor = np.reshape(descriptor, [1, -1])
    return int(model.predict(descriptor)[1].ravel()[0])

def detect_traffic_sign(frame, model, hog):
    """Returns (class_id, bbox) for the largest sign-like candidate, or (None, None)."""
    original = frame.copy()
    binary = sign_preprocess(frame)
    binary = sign_remove_small_components(binary, SIGN_MIN_SIZE_COMPONENTS)
    binary = cv2.bitwise_and(binary, binary, mask=sign_remove_other_color(frame))

    contours = sign_find_contours(binary)
    crop, coordinate = sign_find_largest(original, contours, SIGN_SIMILARITY_THRESHOLD,
                                          SIGN_DISTANCE_THRESHOLD)
    if crop is None or crop.size == 0:
        return None, None

    class_id = sign_classify(model, hog, crop)
    return class_id, coordinate

# Load the pre-trained SVM once at startup. cv2.ml.SVM's *instance* .load() has a
# known bug (https://github.com/opencv/opencv/issues/4969) so we use the module-level
# loader instead.
sign_model = None
sign_hog = sign_get_hog()
try:
    sign_model = cv2.ml.SVM_load(SVM_MODEL_PATH)
    print(f"Loaded traffic sign model from {SVM_MODEL_PATH}")
except Exception as e:
    print(f"Could not load traffic sign model at {SVM_MODEL_PATH}: {e}")
    print("Traffic sign detection will be disabled — obstacle avoidance still runs.")

# ============================================
# PERSISTENCE STATE
# ============================================

last_obstacle = None     # colour-obstacle persistence (unchanged behaviour)
current_forward_speed = FORWARD_SPEED

current_sign_class = None
sign_class_streak = 0
sign_cooldown_until = {}   # class_id -> time until which re-triggers are ignored
active_sign = None         # {"info": ..., "expires_at": ...}

# ================= MAIN SETUP =================
receiver = VideoReceiver(STREAM_PORT)

cv2.namedWindow("Video feed",    cv2.WINDOW_NORMAL)
cv2.namedWindow("Red Mask",      cv2.WINDOW_NORMAL)
cv2.namedWindow("Blue Mask",     cv2.WINDOW_NORMAL)

initialize_board()
set_servo_angle(SERVO_CENTER)
time.sleep(0.5)

# ================= MAIN LOOP =================
while True:
    frame = receiver.get_frame()

    if frame is None:
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for video stream...",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Video feed", placeholder)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        control_motors(0, 0)
        time.sleep(0.05)
        continue

    try:
        frame = cv2.resize(frame, (320, 240))
        h, w = frame.shape[:2]
        debug = frame.copy()
        now = time.time()

        # ---------- 1. Traffic sign detection (runs every frame) ----------
        if sign_model is not None:
            sign_class_id, sign_bbox = detect_traffic_sign(frame, sign_model, sign_hog)
        else:
            sign_class_id, sign_bbox = None, None

        if sign_class_id == current_sign_class:
            sign_class_streak += 1
        else:
            current_sign_class = sign_class_id
            sign_class_streak = 1

        sign_confirmed = (sign_class_id is not None and sign_class_id != 0
                           and sign_class_streak >= SIGN_CONFIRM_FRAMES)

        if sign_confirmed and now >= sign_cooldown_until.get(sign_class_id, 0):
            info = SIGN_INFO.get(sign_class_id)
            if info and info["action"] != "IGNORE":
                active_sign = {"info": info, "expires_at": now + SIGN_ACTION_HOLD_SEC}
                sign_cooldown_until[sign_class_id] = now + SIGN_COOLDOWN_SEC
                print(f"[Sign] Triggered: {info['name']}")

        if active_sign and now >= active_sign["expires_at"]:
            active_sign = None

        if sign_bbox is not None:
            cv2.rectangle(debug, sign_bbox[0], sign_bbox[1], (0, 255, 255), 2)
            if sign_class_id in SIGN_INFO:
                cv2.putText(debug, SIGN_INFO[sign_class_id]["name"],
                            (sign_bbox[0][0], max(0, sign_bbox[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ---------- 2. Colour-based obstacle avoidance (unchanged) ----------
        red_mask  = get_red_mask(frame)
        blue_mask = get_blue_mask(frame)

        red_contours,  _ = cv2.findContours(red_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        decision     = "FORWARD"
        servo_angle  = SERVO_CENTER
        red_found    = False
        blue_found   = False

        valid_red = [c for c in red_contours if cv2.contourArea(c) > 3000]
        if valid_red:
            cnt = max(valid_red, key=cv2.contourArea)
            red_found = True
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cx = x + w_box // 2

            cv2.rectangle(debug, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            cv2.circle(debug, (cx, y + h_box // 2), 5, (0, 255, 255), -1)

            servo_angle = calc_servo_angle(cx, w, "left")
            decision    = f"RED -> LEFT  angle={servo_angle}"

            last_obstacle = {
                "direction": "left",
                "angle":     servo_angle,
                "expires_at": now + OBSTACLE_PERSISTENCE_SEC
            }

        if not red_found:
            valid_blue = [c for c in blue_contours if cv2.contourArea(c) > 3000]
            if valid_blue:
                cnt = max(valid_blue, key=cv2.contourArea)
                blue_found = True
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cx = x + w_box // 2

                cv2.rectangle(debug, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                cv2.circle(debug, (cx, y + h_box // 2), 5, (0, 255, 255), -1)

                servo_angle = calc_servo_angle(cx, w, "right")
                decision    = f"BLUE -> RIGHT  angle={servo_angle}"

                last_obstacle = {
                    "direction": "right",
                    "angle":     servo_angle,
                    "expires_at": now + OBSTACLE_PERSISTENCE_SEC
                }

        if not red_found and not blue_found:
            if last_obstacle and now < last_obstacle["expires_at"]:
                servo_angle = last_obstacle["angle"]
                remaining   = last_obstacle["expires_at"] - now
                decision    = (f"PERSIST {last_obstacle['direction'].upper()} "
                               f"angle={servo_angle} ({remaining:.1f}s left)")
            else:
                last_obstacle = None
                servo_angle   = SERVO_CENTER
                decision      = "FORWARD"

        # ---------- 3. Merge: traffic sign action overrides obstacle avoidance ----------
        drive_speed = current_forward_speed
        stop_robot = False

        if active_sign:
            info = active_sign["info"]
            decision = f"SIGN: {info['name']}"
            if info["action"] == "STOP":
                stop_robot = True
                servo_angle = SERVO_CENTER
            elif info["action"] == "BIAS_LEFT":
                servo_angle = SERVO_LEFT_MAX
            elif info["action"] == "BIAS_RIGHT":
                servo_angle = SERVO_RIGHT_MAX
            elif info["action"] == "SET_SPEED":
                current_forward_speed = info["speed"]
                drive_speed = current_forward_speed

        # ---------- 4. Apply commands ----------
        set_servo_angle(servo_angle)
        if stop_robot:
            control_motors(0, 0)
        else:
            control_motors(drive_speed, drive_speed)

        cv2.putText(debug, decision, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Video feed", debug)
        cv2.imshow("Red Mask",   red_mask)
        cv2.imshow("Blue Mask",  blue_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.070)

    except Exception as ex:
        print("Loop Error:", ex)
        control_motors(0, 0)

# Cleanup
receiver.stop()
control_motors(0, 0)
cv2.destroyAllWindows()
