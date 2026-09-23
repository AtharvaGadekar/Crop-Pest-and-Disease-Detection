#!/usr/bin/env python3
"""Control a Quarky Intellio rover and spray water when a bird is seen.

Connect the computer and Intellio to the same Wi-Fi network, then run:
    python3 quarky_wasd.py --ip INTELLIO_IP --detect-birds

M1 drives with W/S, the servo steers with A/D, and M2 powers the pump.
Bird mode stops the car, sprays for one second, and waits for a clear frame.
"""

import argparse
import ipaddress
import math
import socket
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass


# Network settings copied from the previous controller.
QUARKY_IP = "192.168.137.102"
CONTROL_PORT = 5006

# Intellio sends SIZE/JPEG packets directly to the configured computer on 5005.
DIRECT_VIDEO_HOST = "0.0.0.0"
DIRECT_VIDEO_PORT = 5005
# PictoBlox receives on 5005 and forwards locally to 6000.
PICTOBLOX_VIDEO_HOST = "127.0.0.1"
PICTOBLOX_VIDEO_PORT = 6000
MAX_UDP_PACKET = 65536
MAX_FRAME_SIZE = 20 * 1024 * 1024

# Start slowly. Increase SPEED only after confirming every control works.
SPEED = 100
RIGHT_MOTOR_TRIM = 5
PUMP_SPEED = 100

# The previous code proves that 1 means forward. Its firmware is expected to
# use 2 for reverse; change REVERSE_DIRECTION here if your firmware differs.
FORWARD_DIRECTION = 1
REVERSE_DIRECTION = 2

SERVO_LEFT = 125
SERVO_CENTRE = 90
SERVO_RIGHT = 55


def build_motor_command(left_direction, right_direction, left_speed, right_speed):
    return (
        f"frame/2/201/10/{int(left_direction)}/{int(right_direction)}/"
        f"{int(left_speed)}/{int(right_speed)}"
    )


class UdpSender:
    def __init__(self, host, port):
        self.destination = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, command):
        self.socket.sendto(command.encode("utf-8"), self.destination)

    def close(self):
        self.socket.close()


class VideoFrameAssembler:
    """Assemble the SIZE header and JPEG chunks used by the Intellio stream."""

    def __init__(self, decode_frame):
        self.decode_frame = decode_frame
        self.expected_size = None
        self.buffer = bytearray()

    def reset(self):
        self.expected_size = None
        self.buffer.clear()

    def feed(self, packet):
        if packet.startswith(b"SIZE:"):
            try:
                size = int(packet[5:].strip())
            except (TypeError, ValueError):
                self.reset()
                return None

            if not 0 < size <= MAX_FRAME_SIZE:
                self.reset()
                return None

            self.expected_size = size
            self.buffer.clear()
            return None

        if self.expected_size is None:
            return None

        bytes_needed = self.expected_size - len(self.buffer)
        self.buffer.extend(packet[:bytes_needed])
        if len(self.buffer) < self.expected_size:
            return None

        jpeg = bytes(self.buffer)
        self.reset()
        return self.decode_frame(jpeg)


@dataclass(frozen=True)
class VideoPacket:
    frame_id: int
    jpeg: bytes
    captured_at: float  # Local receive time; the rover sends no capture timestamp.


class VideoReceiver:
    def __init__(self, host, port, decode_frame):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.settimeout(0.25)
        self.assembler = VideoFrameAssembler(decode_frame)
        self.latest_frame = None
        self.latest_packet = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def _receive_loop(self):
        while self.running:
            try:
                packet, _ = self.socket.recvfrom(MAX_UDP_PACKET)
                frame = self.assembler.feed(packet)
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.frame_id += 1
                        self.latest_packet = VideoPacket(self.frame_id, frame, time.monotonic())
            except socket.timeout:
                continue
            except OSError:
                if self.running:
                    raise
                break

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def get_packet(self):
        with self.lock:
            return self.latest_packet

    def stop(self):
        self.running = False
        self.socket.close()
        self.thread.join(timeout=1.0)


def video_bind_address(source):
    if source == "direct":
        return DIRECT_VIDEO_HOST, DIRECT_VIDEO_PORT
    if source == "pictoblox":
        return PICTOBLOX_VIDEO_HOST, PICTOBLOX_VIDEO_PORT
    raise ValueError(f"unknown video source: {source}")


class RobotController:
    def __init__(self, send_command, speed=25, right_trim=5, sleep=time.sleep):
        self.send_command = send_command
        self.speed = max(0, min(100, int(speed)))
        self.right_speed = max(0, min(100, self.speed + int(right_trim)))
        self.sleep = sleep

    def initialize(self):
        self.send_command("frame/2/201/1")
        self.sleep(0.5)
        self.send_command(f"frame/2/33/0/{SERVO_CENTRE}")

    def stop(self):
        self.send_command(
            build_motor_command(FORWARD_DIRECTION, FORWARD_DIRECTION, 0, 0)
        )

    def centre_steering(self):
        self.send_command(f"frame/2/33/0/{SERVO_CENTRE}")

    def safe_idle(self):
        self.stop()
        self.centre_steering()

    def start_pump(self):
        self.safe_idle()
        self.send_command(
            build_motor_command(FORWARD_DIRECTION, FORWARD_DIRECTION, 0, PUMP_SPEED)
        )

    def handle_key_press(self, key):
        key = key.lower()
        if key == "w":
            self.send_command(
                build_motor_command(
                    FORWARD_DIRECTION,
                    FORWARD_DIRECTION,
                    self.speed,
                    0,
                )
            )
        elif key == "s":
            self.send_command(
                build_motor_command(
                    REVERSE_DIRECTION,
                    FORWARD_DIRECTION,
                    self.speed,
                    0,
                )
            )
        elif key == "a":
            self.send_command(f"frame/2/33/0/{SERVO_LEFT}")
        elif key == "d":
            self.send_command(f"frame/2/33/0/{SERVO_RIGHT}")
        elif key == "c":
            self.send_command(f"frame/2/33/0/{SERVO_CENTRE}")
        elif key == " ":
            self.stop()
        elif key == "q":
            self.stop()
            return False
        return True

    def handle_key_release(self, key):
        key = key.lower()
        if key in ("w", "s"):
            self.stop()
        elif key in ("a", "d"):
            self.centre_steering()
        return True


class BirdSprayController:
    """Run one spray per bird sighting and interlock manual driving."""

    def __init__(self, robot, spray_seconds=1.0):
        self.robot = robot
        self.spray_seconds = float(spray_seconds)
        self.spraying = False
        self.drive_enabled = False
        self._bird_latched = False
        self._spray_until = None
        self._failsafe_active = False

    def update(self, snapshot, now=None):
        now = time.monotonic() if now is None else now
        ready = snapshot is not None and snapshot.status == "ready"
        bird_seen = ready and bool(snapshot.detections)

        if not ready:
            if not self._failsafe_active or self.spraying:
                self.robot.safe_idle()
            self.spraying = False
            self._spray_until = None
            self.drive_enabled = False
            self._failsafe_active = True
            return

        self._failsafe_active = False

        if bird_seen and not self._bird_latched:
            self._bird_latched = True
            self.spraying = True
            self._spray_until = now + self.spray_seconds
            self.robot.start_pump()

        if self.spraying and now >= self._spray_until:
            self.robot.stop()
            self.spraying = False

        if ready and not bird_seen and not self.spraying:
            self._bird_latched = False

        self.drive_enabled = ready and not self._bird_latched and not self.spraying


def _draw_frame(screen, jpeg, pygame):
    try:
        surface = pygame.image.load(BytesIO(jpeg), "camera.jpg").convert()
    except pygame.error:
        return

    width, height = surface.get_size()

    screen_width, screen_height = screen.get_size()
    scale = min(screen_width / width, screen_height / height)
    display_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    surface = pygame.transform.smoothscale(surface, display_size)
    position = (
        (screen_width - display_size[0]) // 2,
        (screen_height - display_size[1]) // 2,
    )
    screen.blit(surface, position)
    return pygame.Rect(position, display_size)


def select_bird_display(packet, snapshot, now=None):
    """Use the analyzed JPEG for boxes, and hide video after stream loss."""
    now = time.monotonic() if now is None else now
    if packet is None:
        return None, None, "WAITING FOR VIDEO"
    if now - packet.captured_at > 2.0:
        return None, None, "STREAM LOST"
    if snapshot.status == "ready" and now - snapshot.captured_at <= 1.0:
        count = len(snapshot.detections)
        status = f"{count} DETECTED" if count else "NO BIRD DETECTED"
        return snapshot.jpeg, snapshot, status
    messages = {
        "loading": "LOADING BIRD MODEL",
        "waiting": "WAITING FOR DETECTION",
        "unavailable": "DETECTOR UNAVAILABLE - see terminal",
        "error": "DETECTION ERROR - see terminal",
        "stale": "WAITING FOR FRESH DETECTION",
    }
    return packet.jpeg, None, messages.get(snapshot.status, "WAITING FOR FRESH DETECTION")


def draw_bird_overlay(screen, snapshot, image_rect, font, pygame):
    from bird_detection import scale_box

    image_rect = pygame.Rect(image_rect)
    colour = (70, 230, 140)
    for detection in snapshot.detections:
        rect = pygame.Rect(scale_box(detection.box, (snapshot.width, snapshot.height), image_rect))
        pygame.draw.rect(screen, colour, rect, 2)
        direction = {"LEFT": "< LEFT", "CENTER": "CENTER", "RIGHT": "RIGHT >"}[detection.direction]
        label = font.render(f"{detection.label.upper()} {detection.confidence:.0%} | {direction}", True, colour)
        label_rect = label.get_rect(topleft=(rect.left, max(image_rect.top, rect.top - label.get_height())))
        label_rect.clamp_ip(screen.get_rect())
        pygame.draw.rect(screen, (8, 12, 10), label_rect)
        screen.blit(label, label_rect)


def _draw_bird_status(screen, font, status, snapshot, pygame):
    panel = pygame.Surface((screen.get_width(), 58), pygame.SRCALPHA)
    panel.fill((8, 12, 10, 225))
    panel.blit(font.render(status, True, (235, 235, 235)), (12, 4))
    detail = "Position in camera view | LEFT <40% | CENTER 40-60% | RIGHT >60%"
    if snapshot is not None:
        age_ms = max(0, (time.monotonic() - snapshot.captured_at) * 1000)
        detail = f"Analyzed frame | age {age_ms:.0f} ms | inference {snapshot.inference_ms:.0f} ms"
    small_font = pygame.font.Font(None, 20)
    panel.blit(small_font.render(detail, True, (180, 190, 185)), (12, 32))
    screen.blit(panel, (0, screen.get_height() - panel.get_height()))


def _draw_recognition_overlay(screen, font, snapshot, pygame):
    panel = pygame.Surface((390, 92), pygame.SRCALPHA)
    panel.fill((8, 12, 10, 218))
    if snapshot.status == "unavailable":
        status_colour = (210, 75, 75)
    elif snapshot.status == "ready" and snapshot.confident:
        status_colour = (70, 190, 90)
    else:
        status_colour = (235, 175, 65)

    pygame.draw.rect(panel, status_colour, (0, 0, 7, panel.get_height()))
    heading = font.render("CROP HEALTH RECOGNITION", True, (235, 235, 235))
    result = font.render(snapshot.message, True, status_colour)
    panel.blit(heading, (18, 12))
    panel.blit(result, (18, 49))
    screen.blit(panel, (12, 12))


def run_session(robot, receiver, pygame, inference_worker=None, recognition_error=None,
                bird_worker=None):
    pygame.init()
    screen = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    pygame.display.set_caption("Quarky Intellio — Bird-triggered water spray")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
    last_bird_error = None
    bird_sprayer = BirdSprayController(robot) if bird_worker is not None else None
    key_commands = {
        pygame.K_w: "w",
        pygame.K_a: "a",
        pygame.K_s: "s",
        pygame.K_d: "d",
        pygame.K_c: "c",
        pygame.K_SPACE: " ",
        pygame.K_q: "q",
    }

    try:
        robot.initialize()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    robot.safe_idle()
                elif event.type == pygame.KEYDOWN and event.key in key_commands:
                    key = key_commands[event.key]
                    if (
                        bird_sprayer is None
                        or bird_sprayer.drive_enabled
                        or key in (" ", "q")
                    ):
                        running = robot.handle_key_press(key)
                elif event.type == pygame.KEYUP and event.key in key_commands:
                    robot.handle_key_release(key_commands[event.key])

            screen.fill((18, 18, 18))
            frame = receiver.get_frame()
            bird_snapshot = None
            bird_status = None
            if bird_worker is not None:
                packet = receiver.get_packet()
                if packet is not None and time.monotonic() - packet.captured_at <= 2.0:
                    bird_worker.submit(packet.frame_id, packet.jpeg, packet.captured_at)
                snapshot = bird_worker.snapshot()
                bird_sprayer.update(snapshot)
                if snapshot.error != last_bird_error:
                    last_bird_error = snapshot.error
                    if snapshot.error:
                        print(f"Bird detection: {snapshot.error}", file=sys.stderr)
                frame, bird_snapshot, bird_status = select_bird_display(packet, snapshot)
                if bird_sprayer.spraying:
                    bird_status = "BIRD DETECTED - SPRAYING"
                elif not bird_sprayer.drive_enabled:
                    bird_status = f"{bird_status} - DRIVE LOCKED"
            if frame is None:
                message = font.render(
                    bird_status or "Waiting for Intellio camera stream...",
                    True,
                    (235, 235, 235),
                )
                screen.blit(message, message.get_rect(center=screen.get_rect().center))
            else:
                if inference_worker is not None:
                    inference_worker.submit(frame)
                image_rect = _draw_frame(screen, frame, pygame)
                if image_rect is not None and bird_snapshot is not None:
                    draw_bird_overlay(screen, bird_snapshot, image_rect, font, pygame)

            if inference_worker is not None:
                _draw_recognition_overlay(
                    screen,
                    font,
                    inference_worker.snapshot(),
                    pygame,
                )
            elif recognition_error is not None:
                snapshot = type(
                    "UnavailableSnapshot",
                    (),
                    {
                        "status": "unavailable",
                        "confident": False,
                        "message": "Recognition unavailable",
                    },
                )()
                _draw_recognition_overlay(screen, font, snapshot, pygame)

            if bird_worker is not None:
                _draw_bird_status(screen, font, bird_status, bird_snapshot, pygame)

            pygame.display.flip()
            clock.tick(60)
    finally:
        robot.safe_idle()
        pygame.quit()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Drive Quarky Intellio while viewing its camera stream."
    )
    parser.add_argument(
        "--ip",
        default=QUARKY_IP,
        help=f"Intellio IPv4 address (default: {QUARKY_IP})",
    )
    parser.add_argument(
        "--video-source",
        choices=("direct", "pictoblox"),
        default="direct",
        help="Factory UDP stream or PictoBlox local forwarding (default: direct)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional trained maize pest model artifact",
    )
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=0.5,
        help="Seconds between crop predictions (default: 0.5)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Override the model's confident-result threshold",
    )
    parser.add_argument("--detect-birds", action="store_true",
                        help="Show bird boxes and LEFT/CENTER/RIGHT in the camera view")
    parser.add_argument("--bird-model", type=Path,
                        help="Local YOLO detection model (default: downloads models/yolo11n.pt)")
    parser.add_argument("--bird-confidence", type=float, default=0.5,
                        help="Minimum bird confidence from 0 to 1 (default: 0.5)")
    parser.add_argument("--bird-interval", type=float, default=0.2,
                        help="Minimum seconds between bird predictions (default: 0.2)")
    args = parser.parse_args(argv)
    try:
        ipaddress.IPv4Address(args.ip)
    except ipaddress.AddressValueError:
        parser.error("--ip must be a valid IPv4 address")
    if args.bird_model is not None and not args.detect_birds:
        parser.error("--bird-model requires --detect-birds")
    if not math.isfinite(args.bird_confidence) or not 0 <= args.bird_confidence <= 1:
        parser.error("--bird-confidence must be between 0 and 1")
    if not math.isfinite(args.bird_interval) or args.bird_interval < 0:
        parser.error("--bird-interval must be finite and nonnegative")
    if args.inference_interval < 0:
        parser.error("--inference-interval cannot be negative")
    if (
        args.confidence_threshold is not None
        and not 0.0 <= args.confidence_threshold <= 1.0
    ):
        parser.error("--confidence-threshold must be between 0 and 1")
    return args


def main():
    args = parse_args()
    try:
        import pygame
    except ImportError as error:
        raise SystemExit(
            "Missing video support. Install it with: "
            "python3 -m pip install pygame-ce"
        ) from error

    inference_worker = None
    bird_worker = None
    recognition_error = None
    if args.model is not None:
        try:
            from crop_recognition import InferenceWorker, load_artifact

            classifier = load_artifact(
                args.model,
                confidence_threshold=args.confidence_threshold,
            )
            inference_worker = InferenceWorker(
                classifier,
                interval_seconds=args.inference_interval,
            )
            print(
                f"Recognition model loaded on {classifier.device}: {args.model}"
            )
        except Exception as error:
            recognition_error = str(error)
            print(f"Recognition unavailable: {error}", file=sys.stderr)

    sender = UdpSender(args.ip, CONTROL_PORT)
    robot = RobotController(sender, speed=SPEED, right_trim=RIGHT_MOTOR_TRIM)
    video_host, video_port = video_bind_address(args.video_source)
    stream_url = f"udp://{video_host}:{video_port}"
    try:
        receiver = VideoReceiver(video_host, video_port, lambda jpeg: jpeg)
    except OSError as error:
        sender.close()
        if inference_worker is not None:
            inference_worker.stop()
        hint = " Quit PictoBlox completely and try again." if args.video_source == "direct" else ""
        raise SystemExit(
            f"Could not listen for video on {video_host}:{video_port}: {error}.{hint}"
        ) from error

    print(f"Sending controls to {args.ip}:{CONTROL_PORT}")
    print(f"Receiving camera stream from {stream_url}")
    print("M1 drive: W forward | S reverse")
    print("Servo steering: A left | D right | C centre")
    print("M2 pump: automatic 1-second spray when a bird is detected")
    print("Release to stop/centre | Space stop | Q stop and quit")
    print("Keep this camera window focused while driving.")
    print("Keep the robot lifted off the ground for the first test.")

    try:
        if args.detect_birds:
            from bird_detection import BirdDetectionWorker, DEFAULT_MODEL

            bird_worker = BirdDetectionWorker(
                args.bird_model or DEFAULT_MODEL,
                confidence_threshold=args.bird_confidence,
                interval_seconds=args.bird_interval,
            )
            print("Bird detection enabled. First use downloads YOLO11n; inference runs locally.")
        run_session(
            robot,
            receiver,
            pygame,
            inference_worker=inference_worker,
            recognition_error=recognition_error,
            bird_worker=bird_worker,
        )
    finally:
        if bird_worker is not None:
            bird_worker.stop()
        if inference_worker is not None:
            inference_worker.stop()
        receiver.stop()
        sender.close()
        print("\nController closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
