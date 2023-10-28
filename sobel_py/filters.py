"""Apply filters to an image."""
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def show_rgba(im: np.ndarray, **subplots_kwargs: Any) -> tuple[Figure, Any]:
    """Show the RGBA channels of an image."""
    fig, axs = plt.subplots(2, 2, **subplots_kwargs)
    for i in range(4):
        _im = im[:, :, i]
        ax = axs[i // 2, i % 2]
        _imshow = ax.imshow(_im)
        fig.colorbar(_imshow)
    return fig, axs


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale."""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def blur(a):
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1) * kernel[y, x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum


def sidebyside(img1, img2, **subplots_kwargs: Any):
    """Show two images side by side."""
    fig, axs = plt.subplots(1, 2, **subplots_kwargs)
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    return fig, axs


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
            out_img[i][j] = np.sum(image[i : i + kernel_y, j : j + kernel_y] * kernel)
    return out_img


def gaussian(l=5, sig=1.0):
    """
    Creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def conv_and_show(im: np.ndarray, kernel: np.ndarray, padding: int):
    img_gray = rgb2gray(im)
    out_im = convolution2d(img_gray, kernel=kernel, padding=padding)
    print(img_gray.shape, out_im.shape)
    sidebyside(img_gray, out_im, figsize=(20, 8))
    plt.show()


if __name__ == "__main__":
    filepath = Path("data/pyrs.png")
    im = iio.imread(filepath)

    kernel = np.ones((10, 10))
    norm_kernel = kernel / np.sum(kernel)
    padding = kernel.shape[0] // 2
    conv_and_show(im=im, kernel=norm_kernel, padding=padding)
