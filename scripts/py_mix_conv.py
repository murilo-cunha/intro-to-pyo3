"""Apply the Sobel filter in images."""
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from sobel_py import convolution2d, rgb2gray, sidebyside


def sobel(filepath: Path) -> None:
    """Apply the Sobel filter."""
    img = iio.imread(filepath)
    img_gray = rgb2gray(img)
    sobel_k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    padding = 0

    sobel_arrs = [
        convolution2d(
            img_gray,
            kernel,
            padding=padding,
        )
        for kernel in (sobel_k, sobel_k.T)
    ]
    out_img = np.stack((img_gray,) * 3, axis=-1)
    for i, sobel in enumerate(sobel_arrs):
        _sobel = sobel / sobel.max()
        out_img[..., i] += (_sobel * 255).astype(np.uint8)
    sidebyside(img_gray, out_img)
    plt.show()


if __name__ == "__main__":
    imgpath = Path.cwd().parent / "data/pyrs.png"
    sobel(imgpath)
