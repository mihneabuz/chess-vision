use bytes::Bytes;
use lazy_static::lazy_static;
use reqwest::{
    multipart::{Form, Part},
    Client,
};

lazy_static! {
    static ref CLIENT: Client = Client::new();
    static ref FILE_SERVER: String = crate::utils::file_server();
    static ref FILE_TOKEN: String = crate::utils::file_server_token();
}

pub async fn handle_image(image: Bytes, id: &str) -> Option<String> {
    let image = prepare_image(image)?;
    let hash = format!("{:?}", md5::compute(&image));

    let filename = format!("image_{}", id);
    let url = format!(
        "http://{}/files/{}?token={}",
        FILE_SERVER.as_str(), &filename, FILE_TOKEN.as_str()
    );

    let part = Part::bytes(image).file_name(filename);
    let form = Form::new().part("file", part);

    let res = CLIENT.put(url).multipart(form).send().await;
    res.ok()?.error_for_status().ok()?;

    Some(hash)
}

fn prepare_image(image: Bytes) -> Option<Vec<u8>> {
    Some(image.to_vec())
}
