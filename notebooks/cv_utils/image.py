"""
Module for image processing functions
"""
from typing import List
import cv2
import numpy as np


# def create_img_mask(img):
#     gray_image = np.full((512, 512), 0, dtype=np.uint8)
#     # print(img)
#     img = np.array(img)
#     image_for_mask = img.copy()

#     imgray = cv2.cvtColor(image_for_mask, cv2.COLOR_BGR2GRAY)

#     ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     mask_image = cv2.drawContours(gray_image, contours, -1, (255, 255, 255), 1)

#     mask_image = cv2.resize(mask_image, (512, 512))

#     mask_im_pil = Image.fromarray(mask_image)
#     return mask_im_pil

def make_contour_opencv(image: np.ndarray) -> np.ndarray:
    """
    Baseline contour in GRAY format
    """
    gray_image = np.full((512, 512), 0, dtype=np.uint8)
    # print(img)
    img = np.array(image)
    image_for_mask = img.copy()

    imgray = cv2.cvtColor(image_for_mask, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_image = cv2.drawContours(gray_image, contours, -1, (255, 255, 255), 1)

    mask_image = cv2.resize(mask_image, (512, 512))

    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)

    return mask_image

def make_contour_image(image: np.ndarray) -> np.ndarray:
    """
    Takes rgb image, returns contour in GRAY format
    # TODO: check contour format
    """
    return make_contour_opencv(image)


def make_contours_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Takes list of images returns list of contours
    """
    return [make_contour_image(image) for image in images]
