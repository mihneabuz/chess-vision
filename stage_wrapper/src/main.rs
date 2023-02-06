mod utils;
mod wrapper;

use std::io::Result;

fn main() -> Result<()> {
    let service = wrapper::factory::create(utils::get_service_type())?;

    let image = std::fs::read("../models/boards/data/0.jpg").unwrap();
    let data = vec![(image.clone(), vec![0u8; 1]), (image.clone(), vec![0u8; 1])];
    let results = service.process(&data);

    println!("{:?}", results);

    Ok(())
}
