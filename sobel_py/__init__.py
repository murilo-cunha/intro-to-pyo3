"""Apply filters to images."""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__version__ = "0.1.0"


def show_rgba(im: np.ndarray, **subplots_kwargs: Any) -> tuple[Figure, Any]:
    """Show the RGBA channels of an image."""
    fig, axs = plt.subplots(2, 2, **subplots_kwargs)
    for i in range(4):
        _im = im[:, :, i]
        ax = axs[i // 2, i % 2]
        _imshow = ax.imshow(_im)
        fig.colorbar(_imshow)
    return fig, axs


def sidebyside(
    img_left: np.ndarray,
    img_right: np.ndarray,
    **subplots_kwargs: Any,
) -> tuple[Figure, Any]:
    """Show two images side by side."""
    fig, axs = plt.subplots(1, 2, **subplots_kwargs)
    axs[0].imshow(img_left)
    axs[1].imshow(img_right)
    return fig, axs


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale."""
    return np.dot(
        rgb[..., :3],
        [0.299, 0.587, 0.114],  # https://en.wikipedia.org/wiki/Grayscale
    ).astype(np.uint8)


def convolution2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: int | None = 0,
) -> np.ndarray:
    """2D convolution of an image (stride is 1)."""
    if len(image.shape) != 2:  # noqa: PLR2004
        raise ValueError(f"Expected 2D image, got ({image.shape}).")
    if padding is not None and padding < 0:
        raise ValueError(f"Padding must be non-negative, got {padding}.")
    kernel_y, kernel_x = kernel.shape
    if kernel_y != kernel_x:
        raise ValueError("Expected square kernels, got ({m}, {n}) kernel.")

    padded_image = (
        np.pad(image, (padding, padding), constant_values=0)
        if padding is not None
        else image
    )
    out_img = np.zeros(padded_image.shape)
    img_y, img_x = image.shape
    for i in range(img_y - kernel_y + 1):
        for j in range(img_x - kernel_y + 1):
            out_img[i, j] = np.sum(image[i : i + kernel_y, j : j + kernel_y] * kernel)
    return out_img.astype(np.uint8)
