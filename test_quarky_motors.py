import unittest
from unittest.mock import patch

import quarky_motors
from quarky_motors import MotorController


class MotorControllerTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.controller = MotorController(self.commands.append, speed=40)

    def test_w_drives_only_m1_forward(self):
        self.controller.press("w")
        self.assertEqual(self.commands, ["frame/2/201/10/1/1/40/0"])

    def test_s_drives_only_m1_backward(self):
        self.controller.press("s")
        self.assertEqual(self.commands, ["frame/2/201/10/2/1/40/0"])

    def test_a_drives_only_m2_backward(self):
        self.controller.press("a")
        self.assertEqual(self.commands, ["frame/2/201/10/1/2/0/40"])

    def test_d_drives_only_m2_forward(self):
        self.controller.press("d")
        self.assertEqual(self.commands, ["frame/2/201/10/1/1/0/40"])

    def test_releasing_a_movement_key_stops_both_motors(self):
        self.controller.release("w")
        self.assertEqual(self.commands, ["frame/2/201/10/1/1/0/0"])

    def test_space_stops_both_motors(self):
        self.controller.press(" ")
        self.assertEqual(self.commands, ["frame/2/201/10/1/1/0/0"])

    def test_missing_pygame_recommends_python_314_compatible_package(self):
        real_import = __import__

        def import_without_pygame(name, *args, **kwargs):
            if name == "pygame":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_pygame):
            with self.assertRaisesRegex(SystemExit, "pip install pygame-ce"):
                quarky_motors.main()


if __name__ == "__main__":
    unittest.main()
