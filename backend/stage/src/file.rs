use crate::utils;

use std::io;

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

pub fn fetch_file_sync(filename: &str) -> io::Result<Vec<u8>> {
    let file_server = utils::file_server();
    let token = utils::file_server_token();

    let url = format!("http://{}/files/{}?token={}", file_server, filename, token);

    let response = ureq::get(&url)
        .call()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let mut buf = Vec::with_capacity(4096);
    response.into_reader().read_to_end(&mut buf)?;

    Ok(buf)
}
