use thiserror::Error;

#[derive(Error, Debug)]
pub enum AiError {
    #[error("API request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    
    #[error("Invalid API key")]
    InvalidApiKey,
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Provider error: {0}")]
    ProviderError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Stream error: {0}")]
    StreamError(String),
}

pub type Result<T> = std::result::Result<T, AiError>;