import cv2
import numpy as np

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