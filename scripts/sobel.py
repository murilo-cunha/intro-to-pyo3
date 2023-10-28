"""Apply the Sobel filter in images."""
from sobel_rs import Matrix, convolution2d


def main() -> None:
    matrix = Matrix((2, 2), (1, 2, 3, 4))
    convolution2d(matrix, matrix)
    print("from python", matrix)


if __name__ == "__main__":
    main()
