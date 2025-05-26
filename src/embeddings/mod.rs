pub mod provider;
pub mod openai;
pub mod local;
pub mod models;

pub use provider::{EmbeddingProvider, EmbeddingError};
pub use models::{Embedding, EmbeddingRequest, EmbeddingResponse};
pub use openai::{OpenAIEmbeddingProvider, OpenAIEmbeddingModel};
pub use local::{LocalEmbeddingProvider, MockEmbeddingProvider};