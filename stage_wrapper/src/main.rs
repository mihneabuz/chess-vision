mod message;
mod utils;
mod wrapper;

use futures::join;
use lapin::{Connection, ConnectionProperties, Result};
use message::{consumer, publisher, Message};

#[tokio::main]
async fn main() -> Result<()> {
    println!("starting stage {:?}", utils::model_name());
    println!("connecting to {:?}", utils::amqp_addr());

    let conn = loop {
        if let Ok(conn) = Connection::connect(&utils::amqp_addr(), ConnectionProperties::default()).await {
            break Box::leak(Box::new(conn));
        }

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    };

    println!("connection done");

    let (consume_sender, consume_receiver) = std::sync::mpsc::channel::<Message>();
    let (publish_sender, publish_receiver) = tokio::sync::mpsc::unbounded_channel::<Message>();

    let worker = tokio::task::spawn_blocking(move || {
        let service = wrapper::create(utils::service_type()).expect("could not load service");

        while let Ok(message) = consume_receiver.recv() {
            println!("message: {:?}", message);
            publish_sender.send(Message::new(message.id, Vec::new())).unwrap();
        }
    });

    let h1 = tokio::spawn(consumer(conn, consume_sender));
    let h2 = tokio::spawn(publisher(conn, publish_receiver));

    join! { h1, h2, worker };

    Ok(())
}
