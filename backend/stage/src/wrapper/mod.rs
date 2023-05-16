pub mod python;
pub use python::PythonWrapper;

use std::io::Result;

use crate::utils;

pub trait ServiceWrapper {
    fn resource(&self) -> Result<String>;
    fn configure(&self, data: &[u8]) -> Result<()>;
    fn process(&self, data: &[(&[u8], &[u8])]) -> Result<Vec<Vec<u8>>>;
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
