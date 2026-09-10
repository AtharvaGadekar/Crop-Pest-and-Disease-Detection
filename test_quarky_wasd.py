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
                "frame/2/201/10/1/1/25/30",
                "frame/2/201/10/1/1/0/0",
            ],
        )

    def test_releasing_reverse_key_stops_motors(self):
        self.robot.handle_key_press("s")
        self.robot.handle_key_release("s")

        self.assertEqual(self.commands[-1], "frame/2/201/10/1/1/0/0")

    def test_releasing_steering_key_centres_steering(self):
        self.robot.handle_key_press("a")
        self.robot.handle_key_release("a")

        self.assertEqual(
            self.commands,
            ["frame/2/33/0/55", "frame/2/33/0/90"],
        )


class VideoFrameAssemblerTests(unittest.TestCase):
    def test_returns_jpeg_only_after_all_udp_chunks_arrive(self):
        jpeg = b"\xff\xd8quarky-camera-data\xff\xd9"
        assembler = VideoFrameAssembler(lambda payload: payload)

        self.assertIsNone(assembler.feed(f"SIZE:{len(jpeg)}".encode()))
        self.assertIsNone(assembler.feed(jpeg[:10]))
        decoded = assembler.feed(jpeg[10:])

        self.assertEqual(decoded, jpeg)


if __name__ == "__main__":
    unittest.main()
