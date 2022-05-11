use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::{pymodule, PyErr, PyModule, PyResult, Python};

pub mod search;
pub use search::{binary_search, binary_search_nd};

pub mod hist;
pub use hist::{histnd_parallel, histnd_serial, Weight};

#[pymodule]
fn histnd(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // return an N-dimensional
    // assume bins are sorted
    // maybe check it before calling this function

    // TODO: Generic Data Types
    #[pyfn(m, "histnd_parallel_weighted_f64")]
    fn histnd_parallel_weighted_f64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, f64>,
        py_bins: Vec<PyReadonlyArray1<'py, f64>>,
        py_weights: PyReadonlyArray1<'py, f64>,
        chunksize: usize,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_parallel(
            py_samples.as_array(),
            &bins,
            Weight::HasWeight(py_weights.as_array()),
            chunksize,
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_serial_weighted_f64")]
    fn histnd_serial_weighted_f64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, f64>,
        py_bins: Vec<PyReadonlyArray1<'py, f64>>,
        py_weights: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_serial(
            py_samples.as_array(),
            &bins,
            Weight::HasWeight(py_weights.as_array()),
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_parallel_f64")]
    fn histnd_parallel_f64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, f64>,
        py_bins: Vec<PyReadonlyArray1<'py, f64>>,
        py_weight: f64,
        chunksize: usize,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_parallel(
            py_samples.as_array(),
            &bins,
            Weight::NoWeight(py_weight),
            chunksize,
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_serial_f64")]
    fn histnd_serial_f64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, f64>,
        py_bins: Vec<PyReadonlyArray1<'py, f64>>,
        py_weight: f64,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_serial(py_samples.as_array(), &bins, Weight::NoWeight(py_weight));

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    // TODO: Generic Data Types
    #[pyfn(m, "histnd_parallel_weighted_i64")]
    fn histnd_parallel_weighted_i64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, i64>,
        py_bins: Vec<PyReadonlyArray1<'py, i64>>,
        py_weights: PyReadonlyArray1<'py, f64>,
        chunksize: usize,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_parallel(
            py_samples.as_array(),
            &bins,
            Weight::HasWeight(py_weights.as_array()),
            chunksize,
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_serial_weighted_i64")]
    fn histnd_serial_weighted_i64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, i64>,
        py_bins: Vec<PyReadonlyArray1<'py, i64>>,
        py_weights: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_serial(
            py_samples.as_array(),
            &bins,
            Weight::HasWeight(py_weights.as_array()),
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_parallel_i64")]
    fn histnd_parallel_i64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, i64>,
        py_bins: Vec<PyReadonlyArray1<'py, i64>>,
        py_weight: f64,
        chunksize: usize,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_parallel(
            py_samples.as_array(),
            &bins,
            Weight::NoWeight(py_weight),
            chunksize,
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_serial_i64")]
    fn histnd_serial_i64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, i64>,
        py_bins: Vec<PyReadonlyArray1<'py, i64>>,
        py_weight: f64,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_serial(py_samples.as_array(), &bins, Weight::NoWeight(py_weight));

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    // TODO: Generic Data Types
    #[pyfn(m, "histnd_parallel_weighted_u64")]
    fn histnd_parallel_weighted_u64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, u64>,
        py_bins: Vec<PyReadonlyArray1<'py, u64>>,
        py_weights: PyReadonlyArray1<'py, f64>,
        chunksize: usize,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_parallel(
            py_samples.as_array(),
            &bins,
            Weight::HasWeight(py_weights.as_array()),
            chunksize,
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_serial_weighted_u64")]
    fn histnd_serial_weighted_u64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, u64>,
        py_bins: Vec<PyReadonlyArray1<'py, u64>>,
        py_weights: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_serial(
            py_samples.as_array(),
            &bins,
            Weight::HasWeight(py_weights.as_array()),
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_parallel_u64")]
    fn histnd_parallel_u64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, u64>,
        py_bins: Vec<PyReadonlyArray1<'py, u64>>,
        py_weight: f64,
        chunksize: usize,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_parallel(
            py_samples.as_array(),
            &bins,
            Weight::NoWeight(py_weight),
            chunksize,
        );

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    #[pyfn(m, "histnd_serial_u64")]
    fn histnd_serial_u64_py<'py>(
        py: Python<'py>,
        py_samples: PyReadonlyArray2<'py, u64>,
        py_bins: Vec<PyReadonlyArray1<'py, u64>>,
        py_weight: f64,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let mut bins = Vec::new();
        for elem in py_bins.iter() {
            bins.push(elem.as_array());
        }
        let ret = histnd_serial(py_samples.as_array(), &bins, Weight::NoWeight(py_weight));

        match ret {
            Some(n) => Ok(n.into_pyarray(py)),
            None => Err(PyErr::new::<PyRuntimeError, _>("")),
        }
    }

    Ok(())
}
