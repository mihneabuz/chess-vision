use std::env;

pub fn amqp_addr() -> String {
    format!("amqp://{}:5672", env::var("MESSAGE_BROKER").unwrap_or(String::from("127.0.0.1")))
}

pub fn queue() -> String {
    env::var("QUEUE").expect("QUEUE variable not set")
}
