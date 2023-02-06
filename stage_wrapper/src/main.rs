mod utils;
mod wrapper;

use std::io::Result;

use wrapper::Wrapper;

fn main() -> Result<()> {

    let model = utils::get_model_name();
    let path = utils::get_modules_path();

    let service = Wrapper::new(path, model)?;

    let image = std::fs::read("../models/boards/data/0.jpg").unwrap();
    let data = vec![(image.clone(), vec![0u8; 1]), (image.clone(), vec![0u8; 1])];
    let results = service.process(&data);

    println!("{:?}", results);

    Ok(())
}
