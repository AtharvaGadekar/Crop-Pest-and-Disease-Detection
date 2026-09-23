# Intellio Standalone Connection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure and operate Quarky Intellio without running PictoBlox.

**Architecture:** A focused `intellio_setup.py` module owns USB discovery, credential packet construction, serial transmission, and returned-IP extraction. `quarky_wasd.py` receives Intellio's factory UDP JPEG stream directly on port 5005, while retaining PictoBlox's localhost port-6000 forwarding as an explicit compatibility mode.

**Tech Stack:** Python 3.14 standard library, pyserial for USB serial access, unittest, pygame-ce.

**Spec:** `docs/superpowers/specs/intellio-standalone.md`

## Global Constraints

- Never erase or flash Intellio firmware.
- Use serial port baud rate 115200.
- Preserve all existing motor, pump, and bird-detection fail-safe behavior.
- Default to direct Intellio UDP video on port 5005; PictoBlox forwarding is compatibility-only.

---

### Task 1: USB Wi-Fi Provisioner

**Files:**
- Create: `intellio_setup.py`
- Create: `test_intellio_setup.py`
- Create: `requirements-intellio.txt`

**Interfaces:**
- Produces: `build_wifi_payload(ssid, password, host_ips, connect_type=2) -> bytes`
- Produces: `build_serial_frame(payload, command=201, mode=2) -> bytes`
- Produces: `extract_ipv4(data) -> str | None`
- Produces: `find_serial_ports() -> list[str]`
- Produces: CLI accepting SSID, optional password, optional port, and timeout.

- [x] **Step 1: Write failing packet, validation, IP parsing, and port tests**
- [x] **Step 2: Run `test_intellio_setup.py` and confirm failure because the module is absent**
- [x] **Step 3: Implement the smallest provisioner satisfying the tests**
- [x] **Step 4: Run `test_intellio_setup.py` and confirm all tests pass**
- [x] **Step 5: Install pyserial in `.venv` and perform a read-only port-open check**

### Task 2: Direct Factory UDP Camera Receiver

**Files:**
- Modify: `quarky_wasd.py`
- Modify: `test_quarky_wasd.py`

**Interfaces:**
- Produces: direct binding at `0.0.0.0:5005` using `VideoReceiver`.
- Produces: CLI flags `--ip` and `--video-source {direct,pictoblox}`.

- [x] **Step 1: Write failing source-selection and binding tests**
- [x] **Step 2: Run targeted tests and confirm expected failures**
- [x] **Step 3: Implement direct factory UDP reception and CLI selection**
- [x] **Step 4: Run targeted tests and confirm they pass**

### Task 3: Verification and Hardware Probe

**Files:**
- Modify: `docs/superpowers/plans/2026-09-15-intellio-standalone.md`

**Interfaces:**
- Consumes: standalone provisioner and direct HTTP receiver from Tasks 1 and 2.

- [x] **Step 1: Run the entire unittest suite in `.venv`**
- [x] **Step 2: Run syntax compilation for modified Python files**
- [x] **Step 3: Open the detected Intellio USB port at 115200 without sending credentials**
- [x] **Step 4: Verify a complete JPEG arrives directly from Intellio on UDP port 5005**
