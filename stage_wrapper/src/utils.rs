use std::env;

pub fn get_model_name() -> String {
    env::var("MODELNAME").expect("MODELNAME variable not set")
}

pub fn get_modules_path() -> String {
    env::var("MODULESPATH").unwrap_or(String::from("."))
}
