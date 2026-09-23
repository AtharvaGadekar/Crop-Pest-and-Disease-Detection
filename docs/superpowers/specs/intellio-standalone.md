# Intellio Standalone Connection Specification

## Goal

Run the Quarky Intellio bird-spray controller without opening PictoBlox.

## Requirements

- Keep the factory firmware; do not erase or flash the board.
- Configure a 2.4 GHz router SSID and password over USB serial at 115200 baud.
- Reproduce the credential packet used by the installed PictoBlox application.
- Report the IPv4 address returned by Intellio.
- Receive the camera's factory `SIZE:`/JPEG UDP stream directly on host port 5005.
- Send existing control datagrams directly to `<intellio-ip>:5006`.
- Preserve the bird safety interlock: stop the drive before pumping, spray for one
  second, and keep drive disabled until the bird is no longer detected.
- Keep PictoBlox's localhost port-6000 forwarding as an explicit compatibility mode.

## Safety

- Wi-Fi setup must not invoke `esptool`, erase flash, or write firmware.
- Motor and pump commands remain stopped during connection setup.
- A lost or stale camera/detection stream must keep drive and pump stopped.
