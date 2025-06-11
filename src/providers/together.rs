use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::env;

use crate::{
    CompletionProvider, CompletionRequest, CompletionResponse, StreamChunk,
    Message, MessageContent, Role, Choice, Usage, AiError, Result,
};

/// Together AI provider for various open models
pub struct TogetherProvider {
    client: Client,
    api_key: String,
}

impl TogetherProvider {
    /// Create a new Together AI provider
    /// 
    /// # Arguments
    /// * `api_key` - Optional API key. If not provided, will look for TOGETHER_API_KEY env var
    pub fn new(api_key: Option<String>) -> Result<Self> {
        let api_key = api_key
            .or_else(|| env::var("TOGETHER_API_KEY").ok())
            .ok_or_else(|| AiError::MissingConfiguration {
                field: "api_key".to_string(),
                description: "Together AI API key not provided. Set TOGETHER_API_KEY environment variable or pass it explicitly".to_string(),
            })?;

        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    fn convert_message(&self, message: &Message) -> TogetherMessage {
        let content = match &message.content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Parts(parts) => {
                parts.iter()
                    .filter_map(|part| match part {
                        crate::ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };

        TogetherMessage {
            role: match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            },
            content,
        }
    }

    fn convert_to_standard_response(&self, response: TogetherResponse) -> CompletionResponse {
        CompletionResponse {
            id: response.id,
            model: response.model,
            choices: response.choices.into_iter().map(|choice| Choice {
                index: choice.index,
                message: Message {
                    role: match choice.message.role.as_str() {
                        "system" => Role::System,
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        _ => Role::Assistant,
                    },
                    content: MessageContent::text(choice.message.content),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: choice.finish_reason,
            }).collect(),
            usage: response.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens as u32,
                completion_tokens: u.completion_tokens as u32,
                total_tokens: u.total_tokens as u32,
            }),
        }
    }
}

#[async_trait]
impl CompletionProvider for TogetherProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = "https://api.together.xyz/v1/chat/completions";
        
        let messages: Vec<TogetherMessage> = request.messages
            .iter()
            .map(|msg| self.convert_message(msg))
            .collect();

        let together_request = TogetherChatRequest {
            model: request.model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens.map(|t| t as i32),
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop.clone(),
            stream: false,
            response_format: request.response_format.as_ref().map(|f| TogetherResponseFormat {
                r#type: match &f.r#type {
                    crate::ResponseFormatType::Text => "text",
                    crate::ResponseFormatType::JsonObject => "json_object",
                    crate::ResponseFormatType::JsonSchema => "json_schema",
                }.to_string(),
            }),
        };

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&together_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "together".to_string(),
                message: format!("Together AI API error: {}", error_text),
                error_code: None,
                retryable: status.is_server_error(),
            });
        }

        let together_response: TogetherResponse = response.json().await?;
        Ok(self.convert_to_standard_response(together_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = "https://api.together.xyz/v1/chat/completions";
        
        let messages: Vec<TogetherMessage> = request.messages
            .iter()
            .map(|msg| self.convert_message(msg))
            .collect();

        let together_request = TogetherChatRequest {
            model: request.model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens.map(|t| t as i32),
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop.clone(),
            stream: true,
            response_format: request.response_format.as_ref().map(|f| TogetherResponseFormat {
                r#type: match &f.r#type {
                    crate::ResponseFormatType::Text => "text",
                    crate::ResponseFormatType::JsonObject => "json_object",
                    crate::ResponseFormatType::JsonSchema => "json_schema",
                }.to_string(),
            }),
        };

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&together_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "together".to_string(),
                message: format!("Together AI API error: {}", error_text),
                error_code: None,
                retryable: status.is_server_error(),
            });
        }

        let stream = response.bytes_stream();
        let mapped_stream = stream.map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let text = String::from_utf8_lossy(&chunk);
                    
                    // Together uses server-sent events like OpenAI
                    if let Some(json_str) = text.strip_prefix("data: ") {
                        if json_str.trim() == "[DONE]" {
                            return Ok(StreamChunk {
                                id: "together_stream".to_string(),
                                choices: vec![],
                                model: None,
                            });
                        }
                        
                        match serde_json::from_str::<TogetherStreamResponse>(json_str.trim()) {
                            Ok(together_chunk) => {
                                Ok(StreamChunk {
                                    id: together_chunk.id,
                                    choices: together_chunk.choices.into_iter().map(|choice| crate::StreamChoice {
                                        index: choice.index,
                                        delta: crate::Delta {
                                            role: choice.delta.role.map(|r| match r.as_str() {
                                                "system" => Role::System,
                                                "user" => Role::User,
                                                "assistant" => Role::Assistant,
                                                "tool" => Role::Tool,
                                                _ => Role::Assistant,
                                            }),
                                            content: choice.delta.content,
                                            tool_calls: None,
                                        },
                                        finish_reason: choice.finish_reason,
                                    }).collect(),
                                    model: Some(together_chunk.model),
                                })
                            }
                            Err(e) => Err(AiError::StreamError {
                                message: format!("Failed to parse Together stream chunk: {}", e),
                                retryable: false,
                            }),
                        }
                    } else {
                        // Skip non-data lines
                        Ok(StreamChunk {
                            id: "together_stream".to_string(),
                            choices: vec![],
                            model: None,
                        })
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
        "together"
    }

    fn default_model(&self) -> &'static str {
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            // Mistral models
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            
            // Meta Llama models
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            
            // Qwen models
            "Qwen/Qwen2-72B-Instruct",
            "Qwen/Qwen1.5-72B-Chat",
            
            // DeepSeek models
            "deepseek-ai/deepseek-coder-33b-instruct",
            
            // WizardLM models
            "WizardLM/WizardLM-13B-V1.2",
            
            // Phind models
            "Phind/Phind-CodeLlama-34B-v2",
            
            // NousResearch models
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "NousResearch/Nous-Hermes-2-Yi-34B",
            
            // Code models
            "codellama/CodeLlama-34b-Instruct-hf",
            "codellama/CodeLlama-70b-Instruct-hf",
        ]
    }
}

// Together AI API types

#[derive(Debug, Clone, Serialize)]
struct TogetherChatRequest {
    model: String,
    messages: Vec<TogetherMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<TogetherResponseFormat>,
}

#[derive(Debug, Clone, Serialize)]
struct TogetherMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TogetherMessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct TogetherResponseFormat {
    r#type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TogetherResponse {
    id: String,
    model: String,
    choices: Vec<TogetherChoice>,
    #[serde(default)]
    usage: Option<TogetherUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct TogetherChoice {
    index: u32,
    message: TogetherMessageResponse,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TogetherStreamResponse {
    id: String,
    model: String,
    choices: Vec<TogetherStreamChoice>,
}

#[derive(Debug, Clone, Deserialize)]
struct TogetherStreamChoice {
    index: u32,
    delta: TogetherDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TogetherDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TogetherUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_together_provider_creation() {
        let result = TogetherProvider::new(Some("test-key".to_string()));
        assert!(result.is_ok());
        
        let provider = result.unwrap();
        assert_eq!(provider.name(), "together");
        assert_eq!(provider.default_model(), "mistralai/Mixtral-8x7B-Instruct-v0.1");
    }

    #[test]
    fn test_message_conversion() {
        let provider = TogetherProvider::new(Some("test-key".to_string())).unwrap();
        
        let message = Message {
            role: Role::User,
            content: MessageContent::text("Hello"),
            tool_calls: None,
            tool_call_id: None,
        };
        
        let together_message = provider.convert_message(&message);
        assert_eq!(together_message.role, "user");
        assert_eq!(together_message.content, "Hello");
    }
}