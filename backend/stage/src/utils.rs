use std::env;

pub fn service_type() -> String {
    env::var("SERVICE_TYPE").expect("SERVICE_TYPE variable not set")
}

pub fn model_name() -> String {
    env::var("MODEL_NAME").expect("MODEL_NAME variable not set")
}

pub fn modules_path() -> String {
    env::var("MODULES_PATH").unwrap_or(String::from("."))
}

pub fn amqp_addr() -> String {
    format!("amqp://{}:5672", env::var("MESSAGE_BROKER").unwrap_or(String::from("127.0.0.1")))
}

pub fn current_queue() -> String {
    env::var("CURRENT_QUEUE").expect("CURRENT_QUEUE variable not set")
}

pub fn next_queue() -> String {
    env::var("NEXT_QUEUE").expect("NEXT_QUEUE variable not set")
}

pub fn fail_queue() -> String {
    env::var("FAIL_QUEUE").expect("FAIL_QUEUE variable not set")
}

pub fn file_server() -> String {
    env::var("FILE_SERVER").unwrap_or(String::from("127.0.0.1"))
}

pub fn file_server_token() -> String {
    env::var("FILE_SERVER_TOKEN").unwrap_or(String::from(""))
}
