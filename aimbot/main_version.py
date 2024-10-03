from ultralytics import YOLO
import win32api
import ctypes
import bettercam
import numpy as np

# Load your YOLO model
model = YOLO("models_fp16/LowVideocartFP16.pt").to('cuda')

camera = bettercam.create(max_buffer_len=8)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT)
    ]


def move_mouse(dx, dy, speed_factor=1):
    extra = ctypes.c_ulong(0)
    input_ = INPUT()
    input_.type = INPUT_MOUSE

    scaled_dx = int(dx * speed_factor)
    scaled_dy = int(dy * speed_factor)

    input_.mi = MOUSEINPUT(scaled_dx, scaled_dy, 0,
                           MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))

    ctypes.windll.user32.SendInput(
        1, ctypes.byref(input_), ctypes.sizeof(input_))


def get_cursor_position():
    return win32api.GetCursorPos()


camera.start(target_fps=80)

while True:
    screen = camera.get_latest_frame()

    # conf - Sets the minimum confidence threshold for detections.
    results = model(screen, conf=0.5)[0]

    if len(results) > 0:
        coordinates = results.boxes.xywh.cpu().numpy()
        x, y, w, h = coordinates[0][:4]
        center_x = int(x)
        center_y = int(y)

        current_x, current_y = get_cursor_position()

        dx = int(center_x - current_x)
        dy = int(center_y - current_y - h / 2.5)

        # Adjust the factors as needed to fine-tune the responsiveness
        min_height = 10  # Minimum height of object (close)
        max_height = 200  # Maximum height of object (far)
        min_speed = 0.9  # Minimum speed factor (far)
        max_speed = 1  # Maximum speed factor (close)

        clamped_h = np.clip(h, min_height, max_height)
        distance_factor = (max_height - clamped_h) / \
            (max_height - min_height)
        speed_factor = min_speed + \
            (max_speed - min_speed) * (1 - distance_factor)

        move_mouse(dx, dy, speed_factor)
