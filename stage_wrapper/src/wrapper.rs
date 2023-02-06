use std::io::Result;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyBytes};

pub struct Wrapper {
    model: String,
    service: Py<PyAny>
}

impl Wrapper {
    pub fn new(path: String, model: String) -> Result<Self> {
        pyo3::prepare_freethreaded_python();

        let service = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let syspath: &PyList = py.import("sys")?.getattr("path")?.downcast::<PyList>()?;
            syspath.insert(0, &path)?;

            py.run(&format!("from {} import Service", model), None, None)?;
            let service = py.eval("Service()", None, None)?;

            Ok(service.into())
        })?;

        Ok(Self { model: model, service: service })
    }

    pub fn process(self, data: &[(Vec<u8>, Vec<u8>)]) -> Result<Vec<Vec<u8>>> {
        let results = Python::with_gil(|py| -> PyResult<Vec<Vec<u8>>> {
            let input = PyList::new(py, data.iter().map(|(image, data)| (PyBytes::new(py, image), PyBytes::new(py, data))));
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
}
