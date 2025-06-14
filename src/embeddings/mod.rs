pub mod local;
pub mod models;
pub mod openai;
pub mod provider;

pub use local::{LocalEmbeddingProvider, MockEmbeddingProvider};
pub use models::{Embedding, EmbeddingRequest, EmbeddingResponse};
pub use openai::{OpenAIEmbeddingModel, OpenAIEmbeddingProvider};
pub use provider::{EmbeddingError, EmbeddingProvider};
