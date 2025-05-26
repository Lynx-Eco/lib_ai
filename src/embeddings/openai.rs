use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{
    provider::{EmbeddingProvider, EmbeddingError, Result},
    models::{Embedding, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage},
};

pub struct OpenAIEmbeddingProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }
    
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }
}

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbedding>,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let openai_request = OpenAIEmbeddingRequest {
            input: request.input,
            model: request.model,
        };
        
        let response = self.client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&openai_request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(EmbeddingError::ProviderError(format!("OpenAI API error: {}", error_text)));
        }
        
        let openai_response: OpenAIEmbeddingResponse = response.json().await?;
        
        let embeddings = openai_response.data
            .into_iter()
            .map(|e| Embedding {
                vector: e.embedding,
                index: e.index,
            })
            .collect();
        
        Ok(EmbeddingResponse {
            embeddings,
            usage: Some(EmbeddingUsage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                total_tokens: openai_response.usage.total_tokens,
            }),
        })
    }
    
    fn default_model(&self) -> &str {
        "text-embedding-3-small"
    }
    
    fn dimension(&self) -> usize {
        1536 // dimension for text-embedding-3-small
    }
}

/// Different OpenAI embedding models
pub enum OpenAIEmbeddingModel {
    /// text-embedding-3-small: 1536 dimensions
    TextEmbedding3Small,
    /// text-embedding-3-large: 3072 dimensions
    TextEmbedding3Large,
    /// text-embedding-ada-002: 1536 dimensions (legacy)
    TextEmbeddingAda002,
}

impl OpenAIEmbeddingModel {
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
        }
    }
    
    pub fn dimension(&self) -> usize {
        match self {
            Self::TextEmbedding3Small => 1536,
            Self::TextEmbedding3Large => 3072,
            Self::TextEmbeddingAda002 => 1536,
        }
    }
}