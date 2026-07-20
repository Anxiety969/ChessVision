from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


PROJECT_FOLDER = Path(__file__).parent

SCREENSHOT_PATH = PROJECT_FOLDER / "screenshot.png"

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


def capture_screen():
    screenshot = ImageGrab.grab(all_screens=True)
    screenshot.save(SCREENSHOT_PATH)

    return np.array(screenshot)
def compare_images(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(
        image1_gray,
        image2_gray,
        cv2.TM_CCOEFF_NORMED
    )

    return float(result[0][0])
def load_template(template_name):
    template_path = (
        PROJECT_FOLDER /
        "piece_templates" /
        f"{template_name}.png"
    )

    return cv2.imread(str(template_path))
def identify_piece(square_image, piece_templates):
    best_piece = None
    best_score = -1

    for piece_name, template in piece_templates.items():
        if template is None:
            continue

        score = compare_images(
            square_image,
            template,
        )

        if score > best_score:
            best_score = score
            best_piece = piece_name

    return best_piece, best_score
def square_to_coordinates(square_name):
    file_index = ord(square_name[0]) - ord("a")
    rank_index = 8 - int(square_name[1])

    return rank_index, file_index
def coordinates_to_square(row, column):
    file_name = chr(ord("a") + column)
    rank_name = str(8 - row)

    return f"{file_name}{rank_name}"
def knight_attacks(square_name):
    row, column = square_to_coordinates(square_name)

    moves = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]

    attacked_squares = []

    for row_change, column_change in moves:
        target_row = row + row_change
        target_column = column + column_change

        if 0 <= target_row < 8 and 0 <= target_column < 8:
            attacked_squares.append(
                coordinates_to_square(
                    target_row,
                    target_column,
                )
            )

    return attacked_squares
def king_attacks(square_name):
    row, column = square_to_coordinates(square_name)

    moves = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    attacked_squares = []

    for row_change, column_change in moves:
        target_row = row + row_change
        target_column = column + column_change

        if 0 <= target_row < 8 and 0 <= target_column < 8:
            attacked_squares.append(
                coordinates_to_square(
                    target_row,
                    target_column,
                )
            )

    return attacked_squares
def pawn_attacks(square_name, symbol):
    row, column = square_to_coordinates(square_name)

    row_change = -1 if symbol == "P" else 1

    attacked_squares = []

    for column_change in (-1, 1):
        target_row = row + row_change
        target_column = column + column_change

        if 0 <= target_row < 8 and 0 <= target_column < 8:
            attacked_squares.append(
                coordinates_to_square(
                    target_row,
                    target_column,
                )
            )

    return attacked_squares

def rook_attacks(square_name, occupied_squares):
    directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    return sliding_attacks(
        square_name,
        directions,
        occupied_squares,
    )
    row, column = square_to_coordinates(square_name)

    attacked_squares = []

    directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    for row_change, column_change in directions:
        target_row = row + row_change
        target_column = column + column_change

        while 0 <= target_row < 8 and 0 <= target_column < 8:
            attacked_squares.append(
                coordinates_to_square(
                    target_row,
                    target_column,
                )
            )

            target_row += row_change
            target_column += column_change

    return attacked_squares
def bishop_attacks(square_name, occupied_squares):
    directions = (
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    return sliding_attacks(
        square_name,
        directions,
        occupied_squares,
    )
def queen_attacks(square_name, occupied_squares):
    return rook_attacks(
        square_name,
        occupied_squares,
    ) + bishop_attacks(
        square_name,
        occupied_squares,
    )
def sliding_attacks(square_name, directions, occupied_squares):
    row, column = square_to_coordinates(square_name)

    attacked_squares = []

    for row_change, column_change in directions:
        target_row = row + row_change
        target_column = column + column_change

        while 0 <= target_row < 8 and 0 <= target_column < 8:
            target_square = coordinates_to_square(
                target_row,
                target_column,
            )

            attacked_squares.append(target_square)

            if target_square in occupied_squares:
                break

            target_row += row_change
            target_column += column_change

    return attacked_squares