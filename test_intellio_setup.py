import unittest
from unittest.mock import Mock, patch


class IntellioSetupTests(unittest.TestCase):
    def test_ssid_can_be_prompted_instead_of_exposed_on_command_line(self):
        from intellio_setup import parse_args

        args = parse_args([])

        self.assertIsNone(args.ssid)

    def test_wifi_payload_matches_intellio_router_format(self):
        from intellio_setup import build_wifi_payload

        payload = build_wifi_payload(
            "LabNet",
            "water123",
            ["192.168.1.20", "10.0.0.4"],
        )

        self.assertEqual(
            payload,
            bytes(
                [0, 2, 6, 8]
                + list(b"LabNetwater123")
                + [192, 168, 1, 20, 10, 0, 0, 4]
            ),
        )

    def test_three_character_lengths_use_firmware_sentinel(self):
        from intellio_setup import build_wifi_payload

        payload = build_wifi_payload("Lab", "123", ["192.168.1.20"])

        self.assertEqual(payload[:4], bytes([0, 2, 255, 255]))

    def test_serial_frame_matches_pictoblox_transport_envelope(self):
        from intellio_setup import build_serial_frame

        frame = build_serial_frame(bytes([0, 2, 4, 8, 1, 2, 3]))

        self.assertEqual(
            frame,
            bytes([0xFF, 0x55, 10, 0, 2, 201, 0, 2, 4, 8, 1, 2, 3]),
        )

    def test_rejects_invalid_credentials_and_host_addresses(self):
        from intellio_setup import build_wifi_payload

        invalid = [
            ("", "water123", ["192.168.1.20"]),
            ("LabNet", "", ["192.168.1.20"]),
            ("Láb", "water123", ["192.168.1.20"]),
            ("LabNet", "water123", []),
            ("LabNet", "water123", ["999.1.1.1"]),
        ]
        for ssid, password, addresses in invalid:
            with self.subTest(ssid=ssid, addresses=addresses):
                with self.assertRaises(ValueError):
                    build_wifi_payload(ssid, password, addresses)

    def test_extracts_only_a_valid_ipv4_address_from_serial_noise(self):
        from intellio_setup import extract_ipv4

        self.assertEqual(
            extract_ipv4(b"\xff\x55noise IP=192.168.137.58\r\n"),
            "192.168.137.58",
        )
        self.assertIsNone(extract_ipv4(b"IP=999.168.1.2\r\n"))

    def test_discovers_only_usb_serial_ports(self):
        from intellio_setup import find_serial_ports

        with patch(
            "intellio_setup.glob.glob",
            side_effect=[
                ["/dev/cu.usbserial-10"],
                ["/dev/cu.usbmodem2101"],
                [],
                [],
            ],
        ):
            self.assertEqual(
                find_serial_ports(),
                ["/dev/cu.usbmodem2101", "/dev/cu.usbserial-10"],
            )

    def test_local_ip_falls_back_to_macos_default_interface(self):
        import socket
        from intellio_setup import local_ipv4_addresses

        failed_probe = Mock()
        failed_probe.connect.side_effect = OSError("network probe unavailable")
        with (
            patch("intellio_setup.socket.getaddrinfo", side_effect=socket.gaierror),
            patch("intellio_setup.socket.socket", return_value=failed_probe),
            patch(
                "intellio_setup._macos_default_ipv4",
                return_value="192.168.137.12",
                create=True,
            ),
        ):
            self.assertEqual(local_ipv4_addresses(), ["192.168.137.12"])
        failed_probe.close.assert_called_once()

    def test_reads_ipv4_from_macos_default_interface(self):
        from intellio_setup import _macos_default_ipv4

        with patch(
            "intellio_setup.subprocess.check_output",
            side_effect=[b"   interface: en0\n", b"192.168.137.12\n"],
        ):
            self.assertEqual(_macos_default_ipv4(), "192.168.137.12")


if __name__ == "__main__":
    unittest.main()
