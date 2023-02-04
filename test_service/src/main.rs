mod utils;

use pyo3::prelude::*;
use pyo3::types::PyList;

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    let model = utils::get_model_name();
    let path = utils::get_modules_path();

    Python::with_gil(|py| {
        let syspath: &PyList = py.import("sys")?.getattr("path")?.downcast::<PyList>()?;
        syspath.insert(0, &path)?;

        py.run(&format!("from {} import Service", model), None, None)?;
        let service = py.eval("Service()", None, None)?;

        let name = service.getattr("get_model_name")?.call0()?;
        println!("{}", name);

        Ok(())
    })
}
