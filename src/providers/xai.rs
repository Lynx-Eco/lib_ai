use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;

use crate::{
    providers::openai::OpenAIProvider, CompletionProvider, CompletionRequest, CompletionResponse,
    Result, StreamChunk,
};

pub struct XAIProvider {
    openai_provider: OpenAIProvider,
}

impl XAIProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            openai_provider: OpenAIProvider::with_base_url(
                api_key,
                "https://api.x.ai/v1".to_string(),
            ),
        }
    }
}

#[async_trait]
impl CompletionProvider for XAIProvider {
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
        "xAI"
    }

    fn default_model(&self) -> &'static str {
        "grok-2-latest"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec!["grok-2-latest", "grok-2-1212", "grok-beta"]
    }
}
