#!/usr/bin/env python3
"""Configure Quarky Intellio Wi-Fi over USB without running PictoBlox.

This sends the same runtime command used by PictoBlox. It does not erase,
replace, or update firmware.
"""

import argparse
import getpass
import glob
import ipaddress
import re
import socket
import subprocess
import time


BAUD_RATE = 115200
SERIAL_PATTERNS = (
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/cu.SLAB_USBtoUART*",
    "/dev/cu.wchusbserial*",
)
IPV4_PATTERN = re.compile(rb"(?<!\d)(\d{1,3}(?:\.\d{1,3}){3})(?!\d)")


def _credential_bytes(value, label):
    if not value:
        raise ValueError(f"{label} cannot be empty")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must use ASCII characters") from error
    if len(encoded) > 255:
        raise ValueError(f"{label} is too long")
    return encoded


def build_wifi_payload(ssid, password, host_ips, connect_type=2):
    """Build Intellio's credential payload (2 means connect to a router)."""
    ssid_bytes = _credential_bytes(ssid, "SSID")
    password_bytes = _credential_bytes(password, "Password")
    if connect_type not in (1, 2):
        raise ValueError("connect_type must be 1 (hotspot) or 2 (router)")
    if not host_ips:
        raise ValueError("at least one host IPv4 address is required")

    address_bytes = bytearray()
    for address in host_ips:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as error:
            raise ValueError(f"invalid host IPv4 address: {address}") from error
        if parsed.version != 4:
            raise ValueError(f"host address must be IPv4: {address}")
        address_bytes.extend(parsed.packed)

    # The value 255 for a three-character field is part of Intellio's factory
    # protocol and is intentionally preserved for compatibility.
    ssid_length = 255 if len(ssid_bytes) == 3 else len(ssid_bytes)
    password_length = 255 if len(password_bytes) == 3 else len(password_bytes)
    return bytes(
        [0, connect_type, ssid_length, password_length]
    ) + ssid_bytes + password_bytes + bytes(address_bytes)


def build_serial_frame(payload, command=201, mode=2):
    """Wrap a runtime payload in PictoBlox's FF 55 serial envelope."""
    payload = bytes(payload)
    frame_length = len(payload) + 3
    if frame_length > 255:
        raise ValueError("payload is too large for one Intellio serial frame")
    return bytes([0xFF, 0x55, frame_length, 0, mode, command]) + payload


def extract_ipv4(data):
    """Return the first valid IPv4 address embedded in serial output."""
    for match in IPV4_PATTERN.finditer(bytes(data)):
        candidate = match.group(1).decode("ascii")
        try:
            return str(ipaddress.IPv4Address(candidate))
        except ipaddress.AddressValueError:
            continue
    return None


def find_serial_ports():
    ports = {port for pattern in SERIAL_PATTERNS for port in glob.glob(pattern)}
    return sorted(ports)


def _macos_default_ipv4():
    """Read the address assigned to macOS's default network interface."""
    try:
        route_output = subprocess.check_output(
            ["/sbin/route", "-n", "get", "default"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("ascii", errors="ignore")
        match = re.search(r"^\s*interface:\s*(\S+)\s*$", route_output, re.MULTILINE)
        if match is None:
            return None
        address = subprocess.check_output(
            ["/usr/sbin/ipconfig", "getifaddr", match.group(1)],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("ascii", errors="ignore").strip()
        return str(ipaddress.IPv4Address(address))
    except (OSError, subprocess.SubprocessError, ipaddress.AddressValueError):
        return None


def local_ipv4_addresses():
    """Find usable local addresses for Intellio's return UDP connection."""
    addresses = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(item[4][0])
    except socket.gaierror:
        pass

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()

    macos_address = _macos_default_ipv4()
    if macos_address is not None:
        addresses.add(macos_address)

    return sorted(address for address in addresses if not address.startswith("127."))


def provision_wifi(port, ssid, password, host_ips, timeout=35.0, boot_wait=6.0):
    try:
        import serial
    except ImportError as error:
        raise RuntimeError(
            "pyserial is required; install it with: "
            "python3 -m pip install -r requirements-intellio.txt"
        ) from error

    payload = build_wifi_payload(ssid, password, host_ips)
    frame = build_serial_frame(payload)
    received = bytearray()
    with serial.Serial(
        port,
        BAUD_RATE,
        timeout=0.25,
        write_timeout=2.0,
    ) as connection:
        time.sleep(max(0.0, boot_wait))
        connection.reset_input_buffer()
        connection.write(frame)
        connection.flush()
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            chunk = connection.read(1024)
            if chunk:
                received.extend(chunk)
                board_ip = extract_ipv4(received)
                if board_ip is not None:
                    return board_ip
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Connect Quarky Intellio to Wi-Fi without PictoBlox."
    )
    parser.add_argument(
        "ssid",
        nargs="?",
        help="2.4 GHz router Wi-Fi name (prompted if omitted)",
    )
    parser.add_argument("--password", help="Wi-Fi password (prompted if omitted)")
    parser.add_argument("--port", help="USB serial port; auto-detected if omitted")
    parser.add_argument(
        "--host-ip",
        action="append",
        dest="host_ips",
        help="This Mac's IPv4 address; may be repeated and is auto-detected by default",
    )
    parser.add_argument("--timeout", type=float, default=35.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ports = [args.port] if args.port else find_serial_ports()
    if not ports:
        raise SystemExit("No Intellio USB serial port found. Connect it and try again.")
    if len(ports) > 1 and args.port is None:
        raise SystemExit(
            "More than one USB serial port found; choose one with --port:\n  "
            + "\n  ".join(ports)
        )

    host_ips = args.host_ips or local_ipv4_addresses()
    if not host_ips:
        raise SystemExit("Could not determine this Mac's IPv4 address; use --host-ip.")
    ssid = args.ssid
    if ssid is None:
        ssid = input("2.4 GHz Wi-Fi name (SSID): ").strip()
    password = args.password
    if password is None:
        password = getpass.getpass("Wi-Fi password: ")

    print(f"Configuring Intellio on {ports[0]} at {BAUD_RATE} baud...")
    print(f"Return address candidates: {', '.join(host_ips)}")
    try:
        board_ip = provision_wifi(ports[0], ssid, password, host_ips, timeout=args.timeout)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Setup failed: {error}") from error
    if board_ip is None:
        raise SystemExit(
            "Credentials were sent, but Intellio did not return an IP address. "
            "Confirm this is a 2.4 GHz network and check the password."
        )
    print(f"Intellio connected: {board_ip}")
    print(f"Run controller: python3 quarky_wasd.py --ip {board_ip} --detect-birds")


if __name__ == "__main__":
    main()
