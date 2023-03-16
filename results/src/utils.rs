use std::env;
use std::iter::Map;
use std::time::Duration;

use tokio_retry::strategy::{ExponentialBackoff, jitter};

pub fn amqp_addr() -> String {
    format!("amqp://{}:5672", env::var("MESSAGE_BROKER").unwrap_or(String::from("127.0.0.1")))
}

pub fn queue() -> String {
    env::var("QUEUE").expect("QUEUE variable not set")
}

pub fn retry_strategy() -> Map<ExponentialBackoff, fn(Duration) -> Duration> {
    ExponentialBackoff::from_millis(10).map(jitter)
}
