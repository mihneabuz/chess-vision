use std::env;

pub fn service_type() -> String {
    env::var("SERVICETYPE").expect("SERVICETYPE variable not set")
}

pub fn model_name() -> String {
    env::var("MODELNAME").expect("MODELNAME variable not set")
}

pub fn modules_path() -> String {
    env::var("MODULESPATH").unwrap_or(String::from("."))
}

pub fn amqp_addr() -> String {
    env::var("MESSAGE_BROKER").unwrap_or(String::from("amqp://127.0.0.1:5672"))
}

pub fn current_queue() -> String {
    env::var("CURRENT_QUEUE").expect("CURRENT_QUEUE variable not set")
}

pub fn next_queue() -> String {
    env::var("NEXT_QUEUE").expect("NEXT_QUEUE variable not set")
}
