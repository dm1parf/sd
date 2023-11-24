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

def display_image_tuples(images: List[tuple]) -> None:
    """
    Displays multiple images
    """
    print(list(images[0]))
    display_images_row(list(images[0]))
    # plt.figure(figsize=(20,10))
    # columns = 5
    # images = images[0]
    # for i, image in enumerate(images[0]):
    #     # image = image
    #     plt.subplot(len(images) / columns + 1, columns, i + 1)
    #     plt.imshow(image)
    # for image in images:
    #     plt.figure()
    #     plt.imshow(image)

def display_images_row(images: list) -> None:
    """
    Displays multiple images
    """
    plt.figure(figsize=(20,10))
    columns = 2
    # images = images[0]
    for i, image in enumerate(images):
        # image = image
        print(len(images))
        plt.subplot(1, 2, i + 1)
        plt.imshow(image)
