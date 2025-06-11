use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use futures::stream::Stream;
use std::pin::Pin;

use crate::{CompletionProvider, CompletionRequest, CompletionResponse, StreamChunk, Result, AiError, providers::openai::OpenAIProvider};

pub struct OpenRouterProvider {
    openai_provider: OpenAIProvider,
    client: Client,
    api_key: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: String) -> Self {
        let client = Client::new();
        Self {
            openai_provider: OpenAIProvider::with_base_url(
                api_key.clone(),
                "https://openrouter.ai/api/v1".to_string()
            ),
            client,
            api_key,
        }
    }

    pub async fn list_available_models(&self) -> Result<Vec<OpenRouterModel>> {
        let response = self.client
            .get("https://openrouter.ai/api/v1/models")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError { provider: "openrouter".to_string(), message: format!("OpenRouter API error: {}", error_text), error_code: None, retryable: true });
        }

        let models_response: OpenRouterModelsResponse = response.json().await?;
        Ok(models_response.data)
    }
}

#[derive(Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Deserialize, Clone)]
pub struct OpenRouterModel {
    pub id: String,
    pub name: String,
    pub context_length: u32,
    pub pricing: OpenRouterPricing,
}

#[derive(Deserialize, Clone)]
pub struct OpenRouterPricing {
    pub prompt: String,
    pub completion: String,
}

#[async_trait]
impl CompletionProvider for OpenRouterProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.openai_provider.complete(request).await
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        self.openai_provider.complete_stream(request).await
    }

    fn name(&self) -> &'static str {
        "OpenRouter"
    }

    fn default_model(&self) -> &'static str {
        "anthropic/claude-3-5-sonnet"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            "anthropic/claude-3-5-sonnet",
            "anthropic/claude-3-5-haiku",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "openai/o1",
            "openai/o1-mini",
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-1.5-pro",
            "google/gemini-1.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-large",
            "qwen/qwen-2.5-72b-instruct",
            "deepseek/deepseek-chat",
            "x-ai/grok-2-1212",
        ]
    }
}