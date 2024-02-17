import cv2
import numpy as np
from typing import Tuple, List

def image_overlay(image1, image2, location):
    try:
        h, w = image1.shape[:2]
        h1, w1 = image2.shape[:2]
        x, y = location
        image1[y:y+h1, x:x+w1] = image2
        return image1;
    except cv2.error as err:
        print(err)
def change_image_color(image, origin_color, new_color):
    image[np.where((image==origin_color).all(axis=2))] = new_color
    return image

def locate_image_on_image(locate_image: str, on_image: str, color: Tuple[int, int, int] = (0, 0, 255)):
    try:

        image = cv2.imread(on_image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        template = cv2.imread(locate_image, 0)

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        return max_loc
    except cv2.error as err:
        print(err)