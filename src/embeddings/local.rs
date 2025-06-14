use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{
    models::{Embedding, EmbeddingRequest, EmbeddingResponse},
    provider::{EmbeddingError, EmbeddingProvider, Result},
};

/// Local embedding provider using a REST API (e.g., sentence-transformers server)
pub struct LocalEmbeddingProvider {
    client: Client,
    base_url: String,
    model_name: String,
    dimension: usize,
}

impl LocalEmbeddingProvider {
    /// Create a new local embedding provider
    ///
    /// Expects a server running sentence-transformers or similar
    /// Example: https://github.com/UKPLab/sentence-transformers
    pub fn new(base_url: String, model_name: String, dimension: usize) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model_name,
            dimension,
        }
    }

    /// Create provider for sentence-transformers all-MiniLM-L6-v2
    pub fn all_minilm_l6_v2(base_url: String) -> Self {
        Self::new(base_url, "all-MiniLM-L6-v2".to_string(), 384)
    }

    /// Create provider for sentence-transformers all-mpnet-base-v2
    pub fn all_mpnet_base_v2(base_url: String) -> Self {
        Self::new(base_url, "all-mpnet-base-v2".to_string(), 768)
    }
}

#[derive(Serialize)]
struct LocalEmbeddingRequest {
    texts: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct LocalEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let local_request = LocalEmbeddingRequest {
            texts: request.input,
            model: self.model_name.clone(),
        };

        let response = self
            .client
            .post(format!("{}/embed", self.base_url))
            .json(&local_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(EmbeddingError::ProviderError(format!(
                "Local embedding error: {}",
                error_text
            )));
        }

        let local_response: LocalEmbeddingResponse = response.json().await?;

        let embeddings = local_response
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(index, vector)| Embedding { vector, index })
            .collect();

        Ok(EmbeddingResponse {
            embeddings,
            usage: None, // Local models typically don't report usage
        })
    }

    fn default_model(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Mock embedding provider for testing
pub struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Create a mock provider optimized for similarity testing
    pub fn with_similarity() -> Self {
        Self { dimension: 384 }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let embeddings = request
            .input
            .into_iter()
            .enumerate()
            .map(|(index, text)| {
                // Generate deterministic embeddings based on text hash
                let hash = text.chars().map(|c| c as u32).sum::<u32>();
                let seed = hash as f32 / u32::MAX as f32;

                let vector: Vec<f32> = (0..self.dimension)
                    .map(|i| {
                        let base = seed + (i as f32 / self.dimension as f32);
                        let noise = rng.gen_range(-0.1..0.1);
                        (base + noise).sin()
                    })
                    .collect();

                Embedding { vector, index }
            })
            .collect();

        Ok(EmbeddingResponse {
            embeddings,
            usage: None,
        })
    }

    fn default_model(&self) -> &str {
        "mock-embedding-model"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
