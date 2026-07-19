import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


PROJECT_FOLDER = Path(__file__).parent

SCREENSHOT_PATH = PROJECT_FOLDER / "screenshot.png"
TEMPLATES_FOLDER = PROJECT_FOLDER / "piece_templates"

LIGHT_SQUARE = np.array([235, 236, 208], dtype=np.float32)
DARK_SQUARE = np.array([119, 149, 86], dtype=np.float32)

COLOR_TOLERANCE = 35


STARTING_PIECES = {
    "a8": "black_rook",
    "b8": "black_knight",
    "c8": "black_bishop",
    "d8": "black_queen",
    "e8": "black_king",
    "f8": "black_bishop",
    "g8": "black_knight",
    "h8": "black_rook",

    "a7": "black_pawn",
    "b7": "black_pawn",
    "c7": "black_pawn",
    "d7": "black_pawn",
    "e7": "black_pawn",
    "f7": "black_pawn",
    "g7": "black_pawn",
    "h7": "black_pawn",

    "a2": "white_pawn",
    "b2": "white_pawn",
    "c2": "white_pawn",
    "d2": "white_pawn",
    "e2": "white_pawn",
    "f2": "white_pawn",
    "g2": "white_pawn",
    "h2": "white_pawn",

    "a1": "white_rook",
    "b1": "white_knight",
    "c1": "white_bishop",
    "d1": "white_queen",
    "e1": "white_king",
    "f1": "white_bishop",
    "g1": "white_knight",
    "h1": "white_rook",
}


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


def capture_templates():
    status_label.config(text="Capturing templates...")
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
            candidates.append(
                (width * height, x, y, width, height)
            )

    if not candidates:
        status_label.config(text="No board found.")
        return

    candidates.sort(reverse=True)

    _, board_x, board_y, board_width, board_height = candidates[0]

    board = rgb_image[
        board_y:board_y + board_height,
        board_x:board_x + board_width
    ]

    square_width = board_width / 8
    square_height = board_height / 8

    files = "abcdefgh"
    ranks = "87654321"

    TEMPLATES_FOLDER.mkdir(exist_ok=True)

    saved_names = set()

    for row in range(8):
        for column in range(8):
            square_name = f"{files[column]}{ranks[row]}"

            if square_name not in STARTING_PIECES:
                continue

            piece_name = STARTING_PIECES[square_name]

            if piece_name in saved_names:
                continue

            x1 = round(column * square_width)
            y1 = round(row * square_height)
            x2 = round((column + 1) * square_width)
            y2 = round((row + 1) * square_height)

            square_image = board[y1:y2, x1:x2]

            square_bgr = cv2.cvtColor(
                square_image,
                cv2.COLOR_RGB2BGR
            )

            output_path = (
                TEMPLATES_FOLDER / f"{piece_name}.png"
            )

            cv2.imwrite(
                str(output_path),
                square_bgr
            )

            saved_names.add(piece_name)

    status_label.config(
        text=(
            f"Saved {len(saved_names)} templates.\n"
            "Open the piece_templates folder."
        )
    )


window = tk.Tk()
window.title("ChessVision")
window.geometry("500x270")

title_label = tk.Label(
    window,
    text="ChessVision",
    font=("Arial", 22, "bold")
)
title_label.pack(pady=(25, 8))

instruction_label = tk.Label(
    window,
    text="Leave the board in the starting position.",
    font=("Arial", 11)
)
instruction_label.pack(pady=8)

capture_button = tk.Button(
    window,
    text="Create Piece Templates",
    font=("Arial", 12),
    command=capture_templates,
    padx=18,
    pady=10
)
capture_button.pack(pady=12)

status_label = tk.Label(
    window,
    text="Ready",
    font=("Arial", 10),
    justify="center"
)
status_label.pack(pady=6)

window.mainloop()