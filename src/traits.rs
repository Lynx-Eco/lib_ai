use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;

use crate::{models::*, error::Result};

#[async_trait]
pub trait CompletionProvider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    
    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;
    
    fn name(&self) -> &'static str;
    
    fn default_model(&self) -> &'static str;
    
    fn available_models(&self) -> Vec<&'static str>;
}

#[async_trait]
pub trait ModelProvider {
    fn list_models(&self) -> Vec<ModelInfo>;
    
    fn get_model_info(&self, model_name: &str) -> Option<ModelInfo>;
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub display_name: String,
    pub context_window: u32,
    pub max_output_tokens: u32,
    pub supports_streaming: bool,
    pub supports_functions: bool,
}