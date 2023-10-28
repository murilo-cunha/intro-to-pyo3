use ndarray::s;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
#[derive(Clone)]
struct Matrix {
    shape: (usize, usize),
    data: Vec<u8>,
}

#[pymethods]
impl Matrix {
    #[new]
    fn new(shape: (usize, usize), data: Vec<u8>) -> Self {
        Matrix { shape, data }
    }
}

impl Matrix {
    fn to_ndarray(&self) -> anyhow::Result<ndarray::Array2<u8>> {
        ndarray::Array2::from_shape_vec(self.shape, self.data.clone())
            .map_err(|_| anyhow::anyhow!("Invalid shape"))
    }
}

fn pad_with_zeros(image: ndarray::Array2<u8>, padding: usize) -> ndarray::Array2<u8> {
    let (y, x) = image.dim();
    let mut padded_image = ndarray::Array2::zeros((y + 2 * padding, x + 2 * padding));
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
    fn convolution2d(image: Matrix, kernel: Matrix, padding: Option<u8>) -> PyResult<()> {
        let arr = image.to_ndarray().map_err(|e| {
            PyValueError::new_err("Rust error: ".to_owned() + &e.to_string().to_owned())
        })?;
        println!("from rust {arr:?}");
        Ok(())
    }

    m.add_class::<Matrix>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_with_zeros() {
        let input_img = ndarray::arr2(&[[1, 2], [3, 4]]);
        let expected_img = ndarray::arr2(&[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0],
            [0, 0, 3, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]);
        assert_eq!(pad_with_zeros(input_img, 2), expected_img);
    }
}
