#!/usr/bin/env python3
"""Control a Quarky Intellio rover while viewing its camera stream.

Connect the computer to the Quarky's Wi-Fi network, then run:
    python3 quarky_wasd.py

Controls: W forward, S reverse, A left, D right, C centre, Space stop,
and Q stop/quit. Keep the camera window focused while driving.
"""

import argparse
import socket
import sys
import threading
import time
from io import BytesIO
from pathlib import Path


# Network settings copied from the previous controller.
QUARKY_IP = "192.168.137.199"
CONTROL_PORT = 5006

# PictoBlox forwards the Intellio JPEG stream to this local UDP port.
VIDEO_HOST = "127.0.0.1"
VIDEO_PORT = 6000
MAX_UDP_PACKET = 65536
MAX_FRAME_SIZE = 20 * 1024 * 1024

# Start slowly. Increase SPEED only after confirming every control works.
SPEED = 25
RIGHT_MOTOR_TRIM = 5

# The previous code proves that 1 means forward. Its firmware is expected to
# use 2 for reverse; change REVERSE_DIRECTION here if your firmware differs.
FORWARD_DIRECTION = 1
REVERSE_DIRECTION = 2

SERVO_LEFT = 55
SERVO_CENTRE = 90
SERVO_RIGHT = 125


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


class VideoReceiver:
    def __init__(self, host, port, decode_frame):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.settimeout(0.25)
        self.assembler = VideoFrameAssembler(decode_frame)
        self.latest_frame = None
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
            except socket.timeout:
                continue
            except OSError:
                if self.running:
                    raise
                break

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self.socket.close()
        self.thread.join(timeout=1.0)


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

    def handle_key_press(self, key):
        key = key.lower()
        if key == "w":
            self.send_command(
                build_motor_command(
                    FORWARD_DIRECTION,
                    FORWARD_DIRECTION,
                    self.speed,
                    self.right_speed,
                )
            )
        elif key == "s":
            self.send_command(
                build_motor_command(
                    REVERSE_DIRECTION,
                    REVERSE_DIRECTION,
                    self.speed,
                    self.right_speed,
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


def run_session(robot, receiver, pygame, inference_worker=None, recognition_error=None):
    pygame.init()
    screen = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    pygame.display.set_caption("Quarky Intellio Camera — WASD to drive, Q to quit")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
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
                    running = robot.handle_key_press(key_commands[event.key])
                elif event.type == pygame.KEYUP and event.key in key_commands:
                    robot.handle_key_release(key_commands[event.key])

            screen.fill((18, 18, 18))
            frame = receiver.get_frame()
            if frame is None:
                message = font.render(
                    "Waiting for Intellio camera stream on UDP port 6000...",
                    True,
                    (235, 235, 235),
                )
                screen.blit(message, message.get_rect(center=screen.get_rect().center))
            else:
                if inference_worker is not None:
                    inference_worker.submit(frame)
                _draw_frame(screen, frame, pygame)

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
    args = parser.parse_args(argv)
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

    sender = UdpSender(QUARKY_IP, CONTROL_PORT)
    robot = RobotController(sender, speed=SPEED, right_trim=RIGHT_MOTOR_TRIM)
    try:
        receiver = VideoReceiver(
            VIDEO_HOST,
            VIDEO_PORT,
            lambda jpeg: jpeg,
        )
    except OSError as error:
        sender.close()
        raise SystemExit(
            f"Could not listen for video on {VIDEO_HOST}:{VIDEO_PORT}: {error}"
        ) from error

    print(f"Sending controls to {QUARKY_IP}:{CONTROL_PORT}")
    print(f"Receiving camera stream on {VIDEO_HOST}:{VIDEO_PORT}")
    print("W forward | S reverse | A left | D right | C centre")
    print("Hold W/S to drive | Hold A/D to steer")
    print("Release to stop/centre | Space stop | Q stop and quit")
    print("Keep this camera window focused while driving.")
    print("Keep the robot lifted off the ground for the first test.")

    try:
        run_session(
            robot,
            receiver,
            pygame,
            inference_worker=inference_worker,
            recognition_error=recognition_error,
        )
    finally:
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
