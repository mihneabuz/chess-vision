use std::env;
use uci::{self, EngineError};

pub struct Engine {
    inner: uci::Engine,
}

impl Engine {
    pub fn new() -> Self {
        let engine = uci::Engine::new(&env::var("STOCKFISH").unwrap())
            .unwrap()
            .movetime(400);

        Self { inner: engine }
    }

    pub fn generate(&mut self, fen: String) -> Result<String, EngineError> {
        self.inner.set_position(&fen)?;
        self.inner.bestmove()
    }
}
