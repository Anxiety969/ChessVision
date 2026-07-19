import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


PROJECT_FOLDER = Path(__file__).parent

SCREENSHOT_PATH = PROJECT_FOLDER / "screenshot.png"
BOARD_CROP_PATH = PROJECT_FOLDER / "board_crop.png"
OCCUPANCY_PATH = PROJECT_FOLDER / "board_occupancy.png"


LIGHT_SQUARE = np.array([235, 236, 208], dtype=np.float32)
DARK_SQUARE = np.array([119, 149, 86], dtype=np.float32)

COLOR_TOLERANCE = 35


def create_board_mask(rgb_image):
    image_numbers = rgb_image.astype(np.float32)

    light_difference = image_numbers - LIGHT_SQUARE
    dark_difference = image_numbers - DARK_SQUARE

    light_distance = np.sqrt(
        np.sum(light_difference ** 2, axis=2)
    )

    dark_distance = np.sqrt(
        np.sum(dark_difference ** 2, axis=2)
    )

    light_mask = np.where(
        light_distance <= COLOR_TOLERANCE,
        255,
        0
    ).astype(np.uint8)

    dark_mask = np.where(
        dark_distance <= COLOR_TOLERANCE,
        255,
        0
    ).astype(np.uint8)

    return cv2.bitwise_or(light_mask, dark_mask)


def capture_and_detect_pieces():
    status_label.config(text="Capturing board...")
    window.update_idletasks()

    screenshot = ImageGrab.grab(all_screens=True)
    screenshot.save(SCREENSHOT_PATH)

    rgb_image = np.array(screenshot)

    board_mask = create_board_mask(rgb_image)

    kernel = np.ones((5, 5), dtype=np.uint8)

    cleaned_mask = cv2.morphologyEx(
        board_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if width < 300 or height < 300:
            continue

        aspect_ratio = width / height

        if 0.85 <= aspect_ratio <= 1.15:
            area = width * height
            candidates.append((area, x, y, width, height))

    if not candidates:
        status_label.config(text="No board found.")
        return

    candidates.sort(reverse=True)

    _, x, y, width, height = candidates[0]

    board_crop = rgb_image[
        y:y + height,
        x:x + width
    ]

    board_crop_bgr = cv2.cvtColor(
        board_crop,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(BOARD_CROP_PATH),
        board_crop_bgr
    )

    result_image = board_crop_bgr.copy()

    square_width = width / 8
    square_height = height / 8

    files = "abcdefgh"
    ranks = "87654321"

    occupied_count = 0

    for row in range(8):
        for column in range(8):
            x1 = round(column * square_width)
            y1 = round(row * square_height)
            x2 = round((column + 1) * square_width)
            y2 = round((row + 1) * square_height)

            square_rgb = board_crop[
                y1:y2,
                x1:x2
            ]

            expected_color = (
                LIGHT_SQUARE
                if (row + column) % 2 == 0
                else DARK_SQUARE
            )

            difference = square_rgb.astype(np.float32) - expected_color

            distance = np.sqrt(
                np.sum(difference ** 2, axis=2)
            )

            piece_pixels = np.count_nonzero(
                distance > COLOR_TOLERANCE
            )

            total_pixels = square_rgb.shape[0] * square_rgb.shape[1]

            piece_ratio = piece_pixels / total_pixels

            square_name = f"{files[column]}{ranks[row]}"

            if piece_ratio > 0.18:
                occupied_count += 1
                label = f"{square_name} OCC"
                box_color = (0, 0, 255)
            else:
                label = f"{square_name} EMPTY"
                box_color = (0, 255, 0)

            cv2.rectangle(
                result_image,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )

            cv2.putText(
                result_image,
                label,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1
            )

    cv2.imwrite(
        str(OCCUPANCY_PATH),
        result_image
    )

    status_label.config(
        text=f"Done. Occupied squares: {occupied_count}"
    )


window = tk.Tk()
window.title("ChessVision")
window.geometry("500x260")

title_label = tk.Label(
    window,
    text="ChessVision",
    font=("Arial", 22, "bold")
)
title_label.pack(pady=(25, 8))

instruction_label = tk.Label(
    window,
    text="Keep the full Chess.com board visible.",
    font=("Arial", 11)
)
instruction_label.pack(pady=8)

capture_button = tk.Button(
    window,
    text="Detect Occupied Squares",
    font=("Arial", 12),
    command=capture_and_detect_pieces,
    padx=18,
    pady=10
)
capture_button.pack(pady=12)

status_label = tk.Label(
    window,
    text="Ready",
    font=("Arial", 10)
)
status_label.pack(pady=6)

window.mainloop()