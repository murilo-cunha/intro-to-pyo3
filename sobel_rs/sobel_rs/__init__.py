"""Matrix convolutions for 2D arrays."""
import numpy as np

from sobel_rs.sobel_rs import *  # noqa: F403
from sobel_rs.sobel_rs import Matrix


def np2mat(arr: np.ndarray) -> Matrix:
    """Convert a numpy array to a Matrix."""
    return Matrix(shape=arr.shape, data=arr.flatten().tolist())


def mat2np(mat: Matrix) -> np.ndarray:
    """Convert a Matrix to a numpy array."""
    return np.array(mat.data).reshape(mat.shape)
