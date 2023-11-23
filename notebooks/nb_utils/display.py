"""
Notebook display modules for experiments
"""
from typing import List
from matplotlib import pyplot as plt
import numpy as np

def display_images(images: List[np.ndarray]) -> None:
    """
    Displays multiple images
    """
    for image in images:
        plt.figure()
        plt.imshow(image)
