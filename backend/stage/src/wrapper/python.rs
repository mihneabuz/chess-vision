use std::io::{self, Result};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};

use super::ServiceWrapper;

pub struct PythonWrapper {
    model: String,
    service: Py<PyAny>,
}

impl PythonWrapper {
    pub fn new(model: String) -> Result<Self> {
        pyo3::prepare_freethreaded_python();

        let path = crate::utils::modules_path();
        let service = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let syspath: &PyList = py.import("sys")?.getattr("path")?.downcast::<PyList>()?;
            syspath.insert(0, &path)?;

            py.run(&format!("from {} import Service", model), None, None)?;
            let service = py.eval("Service()", None, None)?;

            Ok(service.into())
        })?;

        Ok(Self { model, service })
    }
}

impl ServiceWrapper for PythonWrapper {
    fn process(&self, data: &[(&[u8], &[u8])]) -> Result<Vec<Vec<u8>>> {
        let results = Python::with_gil(|py| -> PyResult<Vec<Vec<u8>>> {
            let input = PyList::new(py,
                data.iter().map(|(image, data)| (PyBytes::new(py, image), PyBytes::new(py, data))),
            );
            let output = self.service.getattr(py, "process")?.call1(py, (input,))?;

            let mut results = Vec::new();
            for result in output.downcast::<PyList>(py)?.into_iter() {
                let bytes = result.downcast::<PyBytes>()?.extract::<Vec<u8>>()?;
                results.push(bytes);
            }

            Ok(results)
        })?;

        Ok(results)
    }

    fn resource(&self) -> Result<String> {
        Python::with_gil(|py| {
            self.service.getattr(py, "get_model_name")?
                .call0(py)?
                .downcast::<PyString>(py)
                .map(|py_str| String::from(py_str.to_string_lossy()) + "_weights")
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
        })
    }

    fn configure(&self, data: &[u8]) -> Result<()> {
        Python::with_gil(|py| {
            self.service.getattr(py, "load_model")?
                .call1(py, (PyBytes::new(py, data), ))
                .map(|_| ())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
        })
    }
}
