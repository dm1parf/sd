"""
Input output modules for experiments
"""

import os
from typing import List

import cv2
import numpy as np


def read_images_folder(path: str) -> List[np.ndarray]:
    """
    Takes folder and reads all .jpg files in rgb. 
    Returns list
    """
    images_list = list()
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            image_path = os.path.join(path, file)
            images_list.append(read_image_rgb(image_path))
    return images_list


def read_image_rgb(path: str) -> np.ndarray:
    """
    reads img from path returns numpy image in RGB
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
