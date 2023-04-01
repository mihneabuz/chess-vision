use uci::{self, EngineError};
use std::env;

pub struct Engine {
    inner: uci::Engine,
}

impl Engine {
    pub fn new() -> Self {
        let engine = uci::Engine::new(&env::var("STOCKFISH").unwrap()).unwrap();
        Self { inner: engine }
    }

    pub fn generate(&mut self, fen: String) -> Result<String, EngineError> {
        self.inner.set_position(&fen)?;
        self.inner.bestmove()
    }
}
