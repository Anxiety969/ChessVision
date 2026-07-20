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