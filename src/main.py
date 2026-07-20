import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

from vision import (
    capture_screen,
    compare_images,
    create_board_mask,
    identify_piece,
    load_template,
    square_to_coordinates,
    coordinates_to_square,
    knight_attacks,
    king_attacks,
    pawn_attacks,
    rook_attacks,
)   


PROJECT_FOLDER = Path(__file__).parent


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


def capture_templates():
    status_label.config(text="Capturing templates...")
    window.update_idletasks()

    window.withdraw()
    window.update()
    time.sleep(0.4)

    rgb_image = capture_screen()

    window.deiconify()
    window.update()

    board_mask = create_board_mask(rgb_image)

    kernel = np.ones((5, 5), dtype=np.uint8)

    cleaned_mask = cv2.morphologyEx(
        board_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
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
        board_x:board_x + board_width,
    ]
    board_bgr = cv2.cvtColor(
    board,
    cv2.COLOR_RGB2BGR,
)
    square_width = board_width / 8
    square_height = board_height / 8

    files = "abcdefgh"
    ranks = "87654321"

    templates_folder = PROJECT_FOLDER / "piece_templates"
    templates_folder.mkdir(exist_ok=True)

    saved_names = set()

    for row in range(8):
        for column in range(8):
            square_name = f"{files[column]}{ranks[row]}"

            if square_name not in STARTING_PIECES:
                continue

            piece_name = STARTING_PIECES[square_name]
            if piece_name.endswith(("_pawn", "_rook", "_knight", "_bishop")):
                square_color = (
                    "light"
                    if (row + column) % 2 == 0
                    else "dark"
                )

                piece_name = f"{piece_name}_{square_color}"

            if piece_name in saved_names:
                continue

            x1 = round(column * square_width)
            y1 = round(row * square_height)
            x2 = round((column + 1) * square_width)
            y2 = round((row + 1) * square_height)

            square_image = board[y1:y2, x1:x2]

            square_bgr = cv2.cvtColor(
                square_image,
                cv2.COLOR_RGB2BGR,
            )

            output_path = (
                templates_folder /
                f"{piece_name}.png"
            )

            piece_templates[piece_name] = square_bgr

            cv2.imwrite(
                str(output_path),
                square_bgr,
            )

            saved_names.add(piece_name)

    status_label.config(
        text=(
            f"Saved {len(saved_names)} templates.\n"
            "Open the piece_templates folder."
        )
    )

def analyze_board():
    status_label.config(text="Analyzing board...")
    window.update_idletasks()

    window.withdraw()
    window.update()
    time.sleep(0.4)

    rgb_image = capture_screen()

    window.deiconify()
    window.update()

    board_mask = create_board_mask(rgb_image)
    kernel = np.ones((5, 5), dtype=np.uint8)

    cleaned_mask = cv2.morphologyEx(
    board_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2,
)

    print("Board mask cleaned.")
    contours, _ = cv2.findContours(
    cleaned_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

    print(f"Found {len(contours)} contours.")

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

    print(f"Found {len(candidates)} board candidates.")
    if not candidates:
        status_label.config(text="No board found.")
        return

    candidates.sort(reverse=True)

    _, board_x, board_y, board_width, board_height = candidates[0]

    print(
        f"Board found at {board_x}, {board_y} "
        f"with size {board_width}x{board_height}."
    )

    board = rgb_image[
    board_y:board_y + board_height,
    board_x:board_x + board_width,
]

    print("Board cropped for analysis.")

    square_width = board_width / 8
    square_height = board_height / 8

    print(
        f"Each square is approximately "
        f"{square_width:.1f}x{square_height:.1f} pixels."
    )
    files = "abcdefgh"
    ranks = "87654321"

    print("Ready to scan all 64 squares.")
    recognized_position = {}
    piece_symbols = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}
    for row in range(8):
        for column in range(8):
            square_name = f"{files[column]}{ranks[row]}"
            x1 = round(column * square_width)
            y1 = round(row * square_height)
            x2 = round((column + 1) * square_width)
            y2 = round((row + 1) * square_height)

            square_image = board[y1:y2, x1:x2]

            square_bgr = cv2.cvtColor(
                square_image,
                cv2.COLOR_RGB2BGR,
            )

            piece_name, score = identify_piece(
                square_bgr,
                piece_templates,
            )
            if piece_name is not None:
                piece_name = piece_name.replace("_light", "")
                piece_name = piece_name.replace("_dark", "")
            if score < 0.50:
                print(square_name, "empty", f"{score:.3f}")
            else:
                recognized_position[square_name] = piece_symbols[piece_name]

                print(square_name, piece_name, f"{score:.3f}")
    print(f"Recognized {len(recognized_position)} occupied squares.")
    for square_name, symbol in recognized_position.items():
        if symbol.lower() == "n":
            print(
                f"Knight on {square_name} attacks:",
                knight_attacks(square_name),
            )
    for square_name, symbol in recognized_position.items():
        if symbol.lower() == "k":
            print(
                f"King on {square_name} attacks:",
                king_attacks(square_name),
            )
    for square_name, symbol in recognized_position.items():
        if symbol.lower() == "p":
            print(
                f"Pawn on {square_name} attacks:",
                pawn_attacks(square_name, symbol),
            )
    for square_name, symbol in recognized_position.items():
        if symbol.lower() == "r":
            print(
                f"Rook on {square_name} attacks:",
                rook_attacks(square_name),
            )
    for rank in ranks:
        row_symbols = []

        for file in files:
            square_name = f"{file}{rank}"
            symbol = recognized_position.get(square_name, ".")
            row_symbols.append(symbol)

        print(" ".join(row_symbols))
piece_templates = {
"white_pawn_light": load_template("white_pawn_light"),
"white_pawn_dark": load_template("white_pawn_dark"),
"white_knight_light": load_template("white_knight_light"),
"white_knight_dark": load_template("white_knight_dark"),
"white_bishop": load_template("white_bishop"),
"white_rook_light": load_template("white_rook_light"),
"white_rook_dark": load_template("white_rook_dark"),
"white_queen": load_template("white_queen"),
"white_king": load_template("white_king"),
"black_pawn_light": load_template("black_pawn_light"),
"black_pawn_dark": load_template("black_pawn_dark"),
"black_knight_light": load_template("black_knight_light"),
"black_knight_dark": load_template("black_knight_dark"),
"black_bishop": load_template("black_bishop"),
"black_rook_light": load_template("black_rook_light"),
"black_rook_dark": load_template("black_rook_dark"),
"black_queen": load_template("black_queen"),
"black_king": load_template("black_king"),
}


window = tk.Tk()
window.title("ChessVision")
window.geometry("500x360")

title_label = tk.Label(
    window,
    text="ChessVision",
    font=("Arial", 22, "bold"),
)
title_label.pack(pady=(25, 8))

instruction_label = tk.Label(
    window,
    text="Leave the board in the starting position.",
    font=("Arial", 11),
)
instruction_label.pack(pady=8)

capture_button = tk.Button(
    window,
    text="Create Piece Templates",
    font=("Arial", 12),
    command=capture_templates,
    padx=18,
    pady=10,
)
capture_button.pack(pady=12)
analyze_button = tk.Button(
    window,
    text="Analyze Board",
    font=("Arial", 12),
    command=analyze_board,
    padx=18,
    pady=10,
)

analyze_button.pack(pady=8)
status_label = tk.Label(
    window,
    text="Ready",
    font=("Arial", 10),
    justify="center",
)
status_label.pack(pady=6)

window.mainloop()