import unittest

from quarky_wasd import RobotController, VideoFrameAssembler


class RobotControllerKeyReleaseTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.robot = RobotController(
            self.commands.append, speed=25, right_trim=5, sleep=lambda _: None
        )

    def test_releasing_forward_key_stops_motors(self):
        self.robot.handle_key_press("w")
        self.robot.handle_key_release("w")

        self.assertEqual(
            self.commands,
            [
                "frame/2/201/10/1/1/25/0",
                "frame/2/201/10/1/1/0/0",
            ],
        )

    def test_reverse_drives_m1_and_keeps_pump_off(self):
        self.robot.handle_key_press("s")

        self.assertEqual(self.commands, ["frame/2/201/10/2/1/25/0"])

    def test_releasing_reverse_key_stops_motors(self):
        self.robot.handle_key_press("s")
        self.robot.handle_key_release("s")

        self.assertEqual(self.commands[-1], "frame/2/201/10/1/1/0/0")

    def test_releasing_steering_key_centres_steering(self):
        self.robot.handle_key_press("a")
        self.robot.handle_key_release("a")

        self.assertEqual(
            self.commands,
            ["frame/2/33/0/125", "frame/2/33/0/90"],
        )

    def test_start_pump_stops_drive_and_centres_steering_first(self):
        start_pump = getattr(self.robot, "start_pump", None)
        self.assertIsNotNone(start_pump, "RobotController needs start_pump()")

        start_pump()

        self.assertEqual(
            self.commands,
            [
                "frame/2/201/10/1/1/0/0",
                "frame/2/33/0/90",
                "frame/2/201/10/1/1/0/100",
            ],
        )


class VideoFrameAssemblerTests(unittest.TestCase):
    def test_returns_jpeg_only_after_all_udp_chunks_arrive(self):
        jpeg = b"\xff\xd8quarky-camera-data\xff\xd9"
        assembler = VideoFrameAssembler(lambda payload: payload)

        self.assertIsNone(assembler.feed(f"SIZE:{len(jpeg)}".encode()))
        self.assertIsNone(assembler.feed(jpeg[:10]))
        decoded = assembler.feed(jpeg[10:])

        self.assertEqual(decoded, jpeg)


class DirectVideoTests(unittest.TestCase):
    def test_direct_udp_is_default_and_ip_is_configurable(self):
        from quarky_wasd import parse_args

        args = parse_args(["--ip", "192.168.1.50"])

        self.assertEqual(args.ip, "192.168.1.50")
        self.assertEqual(args.video_source, "direct")

    def test_video_source_selects_factory_and_pictoblox_udp_ports(self):
        from quarky_wasd import video_bind_address

        self.assertEqual(video_bind_address("direct"), ("0.0.0.0", 5005))
        self.assertEqual(video_bind_address("pictoblox"), ("127.0.0.1", 6000))

    def test_rejects_invalid_controller_ip(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from quarky_wasd import parse_args

        error_output = StringIO()
        with redirect_stderr(error_output), self.assertRaises(SystemExit):
            parse_args(["--ip", "999.1.2.3"])
        self.assertIn("--ip must be a valid IPv4 address", error_output.getvalue())


class BirdSprayControllerTests(unittest.TestCase):
    @staticmethod
    def snapshot(status="ready", bird=True):
        from types import SimpleNamespace

        detections = (object(),) if bird else ()
        return SimpleNamespace(status=status, detections=detections)

    def test_bird_starts_one_second_spray_then_keeps_drive_locked(self):
        import quarky_wasd as controller

        sprayer_class = getattr(controller, "BirdSprayController", None)
        self.assertIsNotNone(
            sprayer_class, "quarky_wasd needs BirdSprayController"
        )
        commands = []
        robot = controller.RobotController(
            commands.append, speed=25, right_trim=5, sleep=lambda _: None
        )
        sprayer = sprayer_class(robot, spray_seconds=1.0)

        sprayer.update(self.snapshot(bird=True), now=10.0)
        self.assertEqual(
            commands,
            [
                "frame/2/201/10/1/1/0/0",
                "frame/2/33/0/90",
                "frame/2/201/10/1/1/0/100",
            ],
        )
        self.assertTrue(sprayer.spraying)
        self.assertFalse(sprayer.drive_enabled)

        sprayer.update(self.snapshot(bird=True), now=10.999)
        self.assertEqual(len(commands), 3)

        sprayer.update(self.snapshot(bird=True), now=11.0)
        self.assertEqual(commands[-1], "frame/2/201/10/1/1/0/0")
        self.assertFalse(sprayer.spraying)
        self.assertFalse(sprayer.drive_enabled)

    def test_clear_detection_unlocks_drive_and_rearms_next_sighting(self):
        from quarky_wasd import BirdSprayController, RobotController

        commands = []
        robot = RobotController(
            commands.append, speed=25, right_trim=5, sleep=lambda _: None
        )
        sprayer = BirdSprayController(robot, spray_seconds=1.0)

        sprayer.update(self.snapshot(bird=True), now=10.0)
        sprayer.update(self.snapshot(bird=False), now=10.5)
        self.assertFalse(sprayer.drive_enabled)
        sprayer.update(self.snapshot(bird=False), now=11.0)
        self.assertTrue(sprayer.drive_enabled)

        sprayer.update(self.snapshot(bird=True), now=12.0)
        self.assertTrue(sprayer.spraying)
        self.assertEqual(commands.count("frame/2/201/10/1/1/0/100"), 2)

    def test_detection_failure_immediately_stops_pump_and_drive(self):
        from quarky_wasd import BirdSprayController, RobotController

        commands = []
        robot = RobotController(
            commands.append, speed=25, right_trim=5, sleep=lambda _: None
        )
        sprayer = BirdSprayController(robot, spray_seconds=1.0)
        sprayer.update(self.snapshot(bird=True), now=10.0)

        sprayer.update(self.snapshot(status="stale", bird=False), now=10.2)

        self.assertEqual(
            commands[-2:],
            ["frame/2/201/10/1/1/0/0", "frame/2/33/0/90"],
        )
        self.assertFalse(sprayer.spraying)
        self.assertFalse(sprayer.drive_enabled)



class BirdModeIntegrationTests(unittest.TestCase):
    def test_bird_flags_do_not_repurpose_crop_model(self):
        from quarky_wasd import parse_args
        args = parse_args(['--detect-birds', '--bird-model', 'pigeon.pt',
                           '--model', 'crop.pth'])
        self.assertTrue(args.detect_birds)
        self.assertEqual(str(args.bird_model), 'pigeon.pt')
        self.assertEqual(str(args.model), 'crop.pth')

    def test_invalid_bird_options_fail_before_startup(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from quarky_wasd import parse_args
        for flag, value in [('--bird-confidence', 'nan'), ('--bird-confidence', '1.1'),
                            ('--bird-interval', '-1'), ('--bird-interval', 'inf')]:
            with self.subTest(flag=flag, value=value), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(['--detect-birds', flag, value])

    def test_receiver_assigns_id_and_receive_time_to_complete_frames(self):
        import socket
        import time
        from quarky_wasd import VideoReceiver
        self.assertTrue(hasattr(VideoReceiver, 'get_packet'))
        receiver = VideoReceiver('127.0.0.1', 0, lambda jpeg: jpeg)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            destination = receiver.socket.getsockname()
            for expected_id in (1, 2):
                sender.sendto(b'SIZE:4', destination)
                sender.sendto(b'jpeg', destination)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    packet = receiver.get_packet()
                    if packet is not None and packet.frame_id == expected_id:
                        break
                    time.sleep(.005)
                self.assertEqual(packet.frame_id, expected_id)
                self.assertEqual(packet.jpeg, b'jpeg')
                self.assertLess(time.monotonic() - packet.captured_at, 1)
            self.assertEqual(receiver.get_frame(), b'jpeg')
        finally:
            receiver.stop()
            sender.close()

    def test_display_uses_detection_source_frame_and_clears_lost_stream(self):
        from types import SimpleNamespace
        import quarky_wasd as controller
        from bird_detection import DetectionSnapshot
        self.assertTrue(hasattr(controller, 'select_bird_display'))
        packet = SimpleNamespace(jpeg=b'newest', captured_at=10, frame_id=2)
        snapshot = DetectionSnapshot(status='ready', jpeg=b'analyzed', captured_at=9.8,
                                     frame_id=1, width=100, height=80)
        jpeg, shown, status = controller.select_bird_display(packet, snapshot, now=10.1)
        self.assertEqual(jpeg, b'analyzed')
        self.assertIs(shown, snapshot)
        jpeg, shown, status = controller.select_bird_display(packet, snapshot, now=13)
        self.assertIsNone(jpeg)
        self.assertIsNone(shown)
        self.assertEqual(status, 'STREAM LOST')
        jpeg, shown, status = controller.select_bird_display(None, snapshot, now=13)
        self.assertIsNone(shown)
        self.assertIn('WAITING', status)

    def test_resized_overlay_draws_box_at_image_coordinates(self):
        import os
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        import pygame
        import quarky_wasd as controller
        from bird_detection import Detection, DetectionSnapshot
        self.assertTrue(hasattr(controller, 'draw_bird_overlay'))
        pygame.init()
        try:
            screen = pygame.Surface((500, 600))
            font = pygame.font.Font(None, 24)
            snapshot = DetectionSnapshot(status='ready', width=100, height=80,
                detections=(Detection((10, 20, 30, 40), 'bird', .9, 'LEFT'),))
            controller.draw_bird_overlay(screen, snapshot, (0, 100, 500, 400), font, pygame)
            self.assertNotEqual(screen.get_at((50, 250))[:3], (0, 0, 0))
            self.assertEqual(screen.get_at((25, 250))[:3], (0, 0, 0))
        finally:
            pygame.quit()

    def test_quit_remains_responsive_while_slow_inference_interlocks_drive(self):
        import os
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        import pygame
        from bird_detection import BirdDetectionWorker, LoadedDetector
        from quarky_wasd import VideoPacket, run_session
        from test_bird_detection import FakeModel, jpeg_image

        model = FakeModel()
        model.release.clear()
        worker = BirdDetectionWorker(detector_factory=lambda _: LoadedDetector(model, 'cpu'))
        jpeg = jpeg_image()
        packet = VideoPacket(1, jpeg, time.monotonic())
        receiver = SimpleNamespace(get_frame=lambda: jpeg, get_packet=lambda: packet)
        commands = []
        robot = RobotController(commands.append, speed=25, right_trim=5, sleep=lambda _: None)
        try:
            worker.submit(packet.frame_id, jpeg, packet.captured_at)
            self.assertTrue(model.entered.wait(2))
            events = [
                [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)],
                [pygame.event.Event(pygame.KEYUP, key=pygame.K_w)],
                [pygame.event.Event(pygame.QUIT)],
            ]
            with patch.object(pygame.event, 'get', side_effect=events):
                run_session(robot, receiver, pygame, bird_worker=worker)
            self.assertEqual(commands, [
                'frame/2/201/1', 'frame/2/33/0/90',
                'frame/2/201/10/1/1/0/0', 'frame/2/33/0/90',
                'frame/2/201/10/1/1/0/0',
                'frame/2/201/10/1/1/0/0', 'frame/2/33/0/90',
            ])
            # Prediction is still blocked: driving and quitting did not wait for it.
            self.assertFalse(model.release.is_set())
            self.assertNotEqual(worker.snapshot().status, 'ready')
        finally:
            model.release.set()
            worker.stop()

    def test_bird_detection_sprays_pump_and_blocks_manual_drive(self):
        import os
        import time
        from types import SimpleNamespace
        from unittest.mock import patch

        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        import pygame
        from bird_detection import Detection, DetectionSnapshot
        from quarky_wasd import RobotController, VideoPacket, run_session

        jpeg = b'jpeg'
        captured_at = time.monotonic()
        packet = VideoPacket(1, jpeg, captured_at)
        receiver = SimpleNamespace(get_frame=lambda: jpeg, get_packet=lambda: packet)
        snapshot = DetectionSnapshot(
            status='ready', frame_id=1, captured_at=captured_at, jpeg=jpeg,
            width=100, height=80,
            detections=(Detection((10, 10, 30, 30), 'bird', .9, 'LEFT'),),
        )
        worker = SimpleNamespace(submit=lambda *args: None, snapshot=lambda: snapshot)
        commands = []
        robot = RobotController(commands.append, speed=25, right_trim=5,
                                sleep=lambda _: None)
        events = [
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)],
            [pygame.event.Event(pygame.QUIT)],
        ]

        with patch.object(pygame.event, 'get', side_effect=events):
            run_session(robot, receiver, pygame, bird_worker=worker)

        self.assertIn('frame/2/201/10/1/1/0/100', commands)
        self.assertNotIn('frame/2/201/10/1/1/25/0', commands)
        self.assertEqual(
            commands[-2:],
            ['frame/2/201/10/1/1/0/0', 'frame/2/33/0/90'],
        )


if __name__ == "__main__":
    unittest.main()
