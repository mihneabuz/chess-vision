mod message;
mod utils;
mod wrapper;

use futures::join;
use lapin::{Connection, ConnectionProperties, Result};
use message::{consumer, publisher, Message};

#[tokio::main]
async fn main() -> Result<()> {
    let conn = Box::leak(Box::new(
        Connection::connect(&utils::amqp_addr(), ConnectionProperties::default()).await?,
    ));

    let (consume_sender, consume_receiver) = std::sync::mpsc::channel::<Message>();
    let (publish_sender, publish_receiver) = tokio::sync::mpsc::unbounded_channel::<Message>();

    let worker = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        let service = wrapper::factory::create(utils::service_type())?;

        while let Ok(message) = consume_receiver.recv() {
            println!("message: {:?}", message);
            publish_sender.send(Message::new(message.id, Vec::new())).unwrap();
        }

        Ok(())
    });

    let h1 = tokio::spawn(consumer(conn, consume_sender));
    let h2 = tokio::spawn(publisher(conn, publish_receiver));

    join! { h1, h2, worker };

    Ok(())
}
