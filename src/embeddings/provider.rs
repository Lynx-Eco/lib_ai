use async_trait::async_trait;
use thiserror::Error;

use super::models::{Embedding, EmbeddingRequest, EmbeddingResponse};

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for the given texts
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Generate a single embedding
    async fn embed_single(&self, text: &str) -> Result<Embedding> {
        let request = EmbeddingRequest {
            input: vec![text.to_string()],
            model: self.default_model().to_string(),
        };

        let response = self.embed(request).await?;

        response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::ProviderError("No embedding returned".to_string()))
    }

    /// Get the default model for this provider
    fn default_model(&self) -> &str;

    /// Get the embedding dimension for the model
    fn dimension(&self) -> usize;
}
