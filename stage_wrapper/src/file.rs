use crate::utils;

use bytes::Bytes;
use reqwest::Result;

pub async fn fetch_image(id: &str) -> Result<Bytes> {
    let file_server = utils::file_server();
    let token = utils::file_server_token();

    let url = format!("http://{}/files/image_{}?token={}", file_server, id, token);

    let response = reqwest::get(url).await?;
    let content = response.bytes().await?;

    Ok(content)
}
