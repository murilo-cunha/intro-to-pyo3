use ndarray::{s, Array2};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Matrix class to pass 2D arrays between Python and Rust.
/// `shape` defines the dimensions of the 2D matrix.
/// `data` is a flattened 1D array of the matrix.
#[pyclass]
#[derive(Clone)]
struct Matrix {
    #[pyo3(get)]
    shape: (usize, usize),
    #[pyo3(get)]
    data: Vec<f32>,
}

#[pymethods]
impl Matrix {
    #[new]
    fn new(shape: (usize, usize), data: Vec<f32>) -> Self {
        Matrix { shape, data }
    }
}

impl Matrix {
    /// Convert to ndarray.
    fn to_ndarray(&self) -> anyhow::Result<Array2<f32>> {
        Array2::from_shape_vec(self.shape, self.data.clone())
            .map_err(|_| anyhow::anyhow!("Invalid shape"))
    }
    /// Convert from ndarray.
    fn from_ndarray(array: &Array2<f32>) -> Self {
        Matrix::new(array.dim(), array.iter().map(|e| e.to_owned()).collect())
    }
}

impl From<Array2<f32>> for Matrix {
    fn from(array: Array2<f32>) -> Self {
        Matrix::from_ndarray(&array)
    }
}

fn pad_with_zeros(image: &Array2<f32>, padding: usize) -> Array2<f32> {
    let (y, x) = image.dim();
    let mut padded_image = Array2::zeros((y + 2 * padding, x + 2 * padding));
    padded_image
        .slice_mut(s![padding..y + padding, padding..x + padding])
        .assign(&image);
    padded_image
}

/// A Python module implemented in Rust.
#[pymodule]
fn sobel_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Formats the sum of two numbers as string.
    #[pyfn(m)]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    /// 2D convolution of an image (stride is 1).
    #[pyfn(m)]
    fn convolution2d(image: Matrix, kernel: Matrix, padding: Option<f32>) -> PyResult<Matrix> {
        // Convert to ndarray
        let img = image.to_ndarray().map_err(|e| {
            PyValueError::new_err(
                "Error converting `Matrix` to `Array2`: ".to_owned() + &e.to_string().to_owned(),
            )
        })?;
        let k = kernel.to_ndarray().map_err(|e| {
            PyValueError::new_err(
                "Error converting `Matrix` to `Array2`: ".to_owned() + &e.to_string().to_owned(),
            )
        })?;

        // Checks
        if img.ndim() != 2 {
            return Err(PyValueError::new_err("Image must be 2D"));
        }
        let (kernel_y, kernel_x) = k.dim();
        if kernel_y != kernel_x {
            return Err(PyValueError::new_err("Kernel must be square"));
        }

        // Convolution
        let padded_img = pad_with_zeros(&img, padding.unwrap_or(0.) as usize);
        let mut out_img: Array2<f32> = ndarray::Array2::zeros(padded_img.dim());
        let (img_y, img_x) = img.dim();
        for i in 0..(img_y - kernel_y + 1) {
            for j in 0..(img_x - kernel_x + 1) {
                out_img[[i, j]] = (&img.slice(s![i..i + kernel_y, j..j + kernel_x]) * &k).sum();
            }
        }
        Ok(out_img.into())
    }

    m.add_class::<Matrix>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_with_zeros() {
        let input_img = ndarray::arr2(&[[1, 2], [3, 4]]).mapv(|x| x as f32);
        let expected_img = ndarray::arr2(&[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0],
            [0, 0, 3, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        .mapv(|x| x as f32);
        assert_eq!(pad_with_zeros(&input_img, 2), expected_img);
    }
}
