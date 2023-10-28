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
