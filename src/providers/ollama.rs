use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;

use crate::{
    CompletionProvider, CompletionRequest, CompletionResponse, StreamChunk,
    Message, MessageContent, Role, Choice, Usage, AiError, Result,
};

/// Ollama provider for local LLM support
pub struct OllamaProvider {
    client: Client,
    base_url: String,
    #[allow(dead_code)]
    default_model: String,
}

impl OllamaProvider {
    /// Create a new Ollama provider
    /// 
    /// # Arguments
    /// * `base_url` - The base URL for the Ollama API (default: "http://localhost:11434")
    /// * `default_model` - The default model to use (e.g., "llama2", "mistral", "codellama")
    pub fn new(base_url: Option<String>, default_model: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            default_model: default_model.unwrap_or_else(|| "llama2".to_string()),
        }
    }

    /// List available models on the Ollama server
    pub async fn list_models(&self) -> Result<Vec<OllamaModel>> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "ollama".to_string(),
                message: format!("Failed to list models: {}", error_text),
                error_code: None,
                retryable: false,
            });
        }
        
        let models_response: OllamaModelsResponse = response.json().await?;
        Ok(models_response.models)
    }

    /// Pull a model from the Ollama registry
    pub async fn pull_model(&self, model_name: &str) -> Result<()> {
        let url = format!("{}/api/pull", self.base_url);
        let request = OllamaPullRequest {
            name: model_name.to_string(),
            stream: false,
        };
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "ollama".to_string(),
                message: format!("Failed to pull model {}: {}", model_name, error_text),
                error_code: None,
                retryable: true,
            });
        }
        
        Ok(())
    }

    /// Check if Ollama is running and accessible
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    fn convert_message(&self, message: &Message) -> OllamaMessage {
        let content = match &message.content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Parts(parts) => {
                // For multimodal, we'll need to handle images differently
                // For now, just extract text parts
                parts.iter()
                    .filter_map(|part| match part {
                        crate::ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };

        OllamaMessage {
            role: match message.role {
                Role::System => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
                Role::Tool => "assistant".to_string(), // Ollama doesn't have a specific tool role
            },
            content,
            images: None, // TODO: Extract images from multimodal content
        }
    }

    fn convert_to_standard_response(&self, response: OllamaResponse) -> CompletionResponse {
        CompletionResponse {
            id: response.created_at.clone().unwrap_or_else(|| "ollama_response".to_string()),
            model: response.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::text(response.message.content),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: if response.done { Some("stop".to_string()) } else { None },
            }],
            usage: Some(Usage {
                prompt_tokens: response.prompt_eval_count.unwrap_or(0) as u32,
                completion_tokens: response.eval_count.unwrap_or(0) as u32,
                total_tokens: (response.prompt_eval_count.unwrap_or(0) + response.eval_count.unwrap_or(0)) as u32,
            }),
        }
    }
}

#[async_trait]
impl CompletionProvider for OllamaProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = format!("{}/api/chat", self.base_url);
        
        // Convert messages
        let messages: Vec<OllamaMessage> = request.messages
            .iter()
            .map(|msg| self.convert_message(msg))
            .collect();

        // Build Ollama request
        let ollama_request = OllamaChatRequest {
            model: request.model.clone(),
            messages,
            stream: false,
            format: request.response_format.as_ref().and_then(|f| {
                match &f.r#type {
                    crate::ResponseFormatType::JsonObject => Some("json".to_string()),
                    _ => None,
                }
            }),
            options: OllamaOptions {
                temperature: request.temperature,
                top_p: request.top_p,
                seed: None,
                num_predict: request.max_tokens.map(|t| t as i32),
                stop: request.stop.clone(),
            },
        };

        let response = self.client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "ollama".to_string(),
                message: format!("Ollama API error: {}", error_text),
                error_code: None,
                retryable: status.is_server_error(),
            });
        }

        let ollama_response: OllamaResponse = response.json().await?;
        Ok(self.convert_to_standard_response(ollama_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = format!("{}/api/chat", self.base_url);
        
        // Convert messages
        let messages: Vec<OllamaMessage> = request.messages
            .iter()
            .map(|msg| self.convert_message(msg))
            .collect();

        // Build Ollama request with streaming enabled
        let ollama_request = OllamaChatRequest {
            model: request.model.clone(),
            messages,
            stream: true,
            format: request.response_format.as_ref().and_then(|f| {
                match &f.r#type {
                    crate::ResponseFormatType::JsonObject => Some("json".to_string()),
                    _ => None,
                }
            }),
            options: OllamaOptions {
                temperature: request.temperature,
                top_p: request.top_p,
                seed: None,
                num_predict: request.max_tokens.map(|t| t as i32),
                stop: request.stop.clone(),
            },
        };

        let response = self.client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "ollama".to_string(),
                message: format!("Ollama API error: {}", error_text),
                error_code: None,
                retryable: status.is_server_error(),
            });
        }

        // Convert the response stream
        let stream = response.bytes_stream();
        let mapped_stream = stream.map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // Parse the JSON line
                    match serde_json::from_slice::<OllamaStreamResponse>(&chunk) {
                        Ok(ollama_chunk) => {
                            Ok(StreamChunk {
                                id: "ollama_stream".to_string(),
                                choices: vec![crate::StreamChoice {
                                    index: 0,
                                    delta: crate::Delta {
                                        role: if ollama_chunk.message.role.is_empty() { 
                                            None 
                                        } else { 
                                            Some(Role::Assistant) 
                                        },
                                        content: if ollama_chunk.message.content.is_empty() { 
                                            None 
                                        } else { 
                                            Some(ollama_chunk.message.content) 
                                        },
                                        tool_calls: None,
                                    },
                                    finish_reason: if ollama_chunk.done { 
                                        Some("stop".to_string()) 
                                    } else { 
                                        None 
                                    },
                                }],
                                model: Some(ollama_chunk.model),
                            })
                        }
                        Err(e) => Err(AiError::StreamError {
                            message: format!("Failed to parse Ollama stream chunk: {}", e),
                            retryable: false,
                        }),
                    }
                }
                Err(e) => Err(AiError::StreamError {
                    message: e.to_string(),
                    retryable: true,
                }),
            }
        });

        Ok(Box::pin(mapped_stream))
    }

    fn name(&self) -> &'static str {
        "ollama"
    }

    fn default_model(&self) -> &'static str {
        "llama2"
    }

    fn available_models(&self) -> Vec<&'static str> {
        // These are commonly available models in Ollama
        // The actual available models depend on what's pulled locally
        vec![
            "llama2",
            "llama2:13b",
            "llama2:70b",
            "mistral",
            "mixtral",
            "codellama",
            "phi",
            "neural-chat",
            "starling-lm",
            "orca-mini",
            "vicuna",
            "gemma",
            "dolphin-mistral",
        ]
    }
}

// Ollama API types

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>, // Base64 encoded images
}

#[derive(Debug, Clone, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    options: OllamaOptions,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaResponse {
    model: String,
    created_at: Option<String>,
    message: OllamaMessage,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaStreamResponse {
    model: String,
    #[allow(dead_code)]
    created_at: Option<String>,
    message: OllamaMessage,
    done: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaPullRequest {
    name: String,
    stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaModelsResponse {
    models: Vec<OllamaModel>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provider_creation() {
        let provider = OllamaProvider::new(None, None);
        assert_eq!(provider.name(), "ollama");
        assert_eq!(provider.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_custom_base_url() {
        let provider = OllamaProvider::new(Some("http://custom:11434".to_string()), Some("mistral".to_string()));
        assert_eq!(provider.base_url, "http://custom:11434");
        assert_eq!(provider.default_model(), "mistral");
    }
}