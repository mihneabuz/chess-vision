pub mod python;
pub use python::PythonWrapper;

use std::io::Result;

use crate::utils;

pub trait ServiceWrapper {
    fn process(&self, data: &[(Vec<u8>, Vec<u8>)]) -> std::io::Result<Vec<Vec<u8>>>;
}

pub fn create(service_type: String) -> Result<Box<dyn ServiceWrapper>> {
    Ok(match service_type.as_str() {
        "python" => {
            let model = utils::model_name();
            Box::new(PythonWrapper::new(model)?)
        }
        _ => panic!("unknown service type"),
    })
}
