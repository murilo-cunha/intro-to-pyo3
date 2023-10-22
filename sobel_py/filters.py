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
    fig, axs = plt.subplots(1, 2, **subplots_kwargs)
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    return fig, axs


def convolution2d(image, kernel):
    m, n = kernel.shape
    if m == n:
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i : i + m, j : j + m] * kernel)
    return new_image


def gaussian(l=5, sig=1.0):
    """
    Creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


if __name__ == "__main__":
    filepath = Path("data/pyrs.png")
    im = iio.imread(filepath)
    # im = iio.imread(
    #     "https://images.unsplash.com/photo-1507786288065-5886035b4d98?auto=format&fit=crop&q=80&w=2278&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    # )
    # fig, _ = show_rgba(im, figsize=(10, 8))
    # fig.suptitle(f"{filepath} ({'x'.join(str(d) for d in im.shape)})")
    # plt.show()

    # plt.imshow(im[200:600, 200:600, :])
    # plt.show()
    _np = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    blur_im = convolution2d(rgb2gray(im), gaussian(5))
    # plt.imshow(blur_im)

    sidebyside(rgb2gray(im), blur_im, figsize=(20, 8))
    plt.show()
