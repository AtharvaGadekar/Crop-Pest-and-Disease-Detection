#!/usr/bin/env python3
"""Control motors connected to Quarky M1 and M2 with W/A/S/D."""

import socket


# Change these settings if your Quarky uses a different address or speed.
QUARKY_IP = "192.168.137.58"
CONTROL_PORT = 5006
SPEED = 40  # 0 to 100

FORWARD = 1
BACKWARD = 2


def motor_command(m1_direction, m2_direction, m1_speed, m2_speed):
    return (
        f"frame/2/201/10/{m1_direction}/{m2_direction}/"
        f"{m1_speed}/{m2_speed}"
    )


class MotorController:
    def __init__(self, send, speed=SPEED):
        self.send = send
        self.speed = max(0, min(100, int(speed)))

    def stop(self):
        self.send(motor_command(FORWARD, FORWARD, 0, 0))

    def press(self, key):
        movements = {
            "w": (FORWARD, FORWARD, self.speed, 0),
            "s": (BACKWARD, FORWARD, self.speed, 0),
            "a": (FORWARD, BACKWARD, 0, self.speed),
            "d": (FORWARD, FORWARD, 0, self.speed),
        }
        if key in movements:
            self.send(motor_command(*movements[key]))
        elif key == " ":
            self.stop()

    def release(self, key):
        if key in "wasd":
            self.stop()


def main():
    try:
        import pygame
    except ImportError:
        raise SystemExit("Install pygame first: python3 -m pip install pygame-ce")

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(command):
        udp_socket.sendto(command.encode("utf-8"), (QUARKY_IP, CONTROL_PORT))

    controller = MotorController(send)
    pygame.init()
    screen = pygame.display.set_mode((460, 150))
    pygame.display.set_caption("Quarky M1/M2 Controller")
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()

    # Enable the Quarky extension before sending motor commands.
    send("frame/2/201/1")

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
                    if key == "q":
                        running = False
                    else:
                        controller.press(key)
                elif event.type == pygame.KEYUP:
                    controller.release(pygame.key.name(event.key))

            screen.fill((25, 28, 35))
            line1 = font.render("W/S: M1     A/D: M2", True, (240, 240, 240))
            line2 = font.render("Space: stop     Q: quit", True, (240, 240, 240))
            screen.blit(line1, (70, 38))
            screen.blit(line2, (70, 82))
            pygame.display.flip()
            clock.tick(60)
    finally:
        controller.stop()
        udp_socket.close()
        pygame.quit()


if __name__ == "__main__":
    main()
