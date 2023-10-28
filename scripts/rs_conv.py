"""Apply the Sobel filter in images."""
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from sobel_py import rgb2gray, sidebyside
from sobel_rs import Matrix, convolution2d


def np2mat(arr: np.ndarray) -> Matrix:
    """Convert a numpy array to a Matrix."""
    return Matrix(shape=arr.shape, data=arr.flatten().tolist())


def mat2np(mat: Matrix) -> np.ndarray:
    """Convert a Matrix to a numpy array."""
    return np.array(mat.data).reshape(mat.shape)


def transform(filepath: Path) -> None:
    """Transform image."""
    img = iio.imread(filepath)
    img_gray = rgb2gray(img)
    kernel = np.ones((10, 10))
    kernel = kernel / np.sum(kernel)
    padding = kernel.shape[0] // 2
    img_out = convolution2d(
        np2mat(img_gray),
        np2mat(kernel),
        padding=padding,
    )
    sidebyside(img_gray, mat2np(img_out).astype(np.uint8))
    plt.show()


def sobel(filepath: Path) -> None:
    """Apply the Sobel filter."""
    img = iio.imread(filepath)
    img_gray = rgb2gray(img)
    sobel_k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    padding = 0

    sobel_mats = [
        convolution2d(
            np2mat(img_gray),
            np2mat(kernel),
            padding=padding,
        )
        for kernel in (sobel_k, sobel_k.T)
    ]
    out_img = np.stack((img_gray,) * 3, axis=-1)
    sobel_arrs = [mat2np(img_out) for img_out in sobel_mats]
    for i, sobel in enumerate(sobel_arrs):
        _sobel = sobel / sobel.max()
        out_img[..., i] += (_sobel * 255).astype(np.uint8)
    print(out_img[300:450, 200:400, :])
    sidebyside(img_gray, out_img)
    plt.show()


if __name__ == "__main__":
    imgpath = Path.cwd().parent / "data/pyrs.png"
    transform(imgpath)
    sobel(imgpath)
