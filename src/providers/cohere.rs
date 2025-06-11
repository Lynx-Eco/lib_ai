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

/// Cohere provider for their AI models
pub struct CohereProvider {
    client: Client,
    api_key: String,
}

impl CohereProvider {
    /// Create a new Cohere provider
    /// 
    /// # Arguments
    /// * `api_key` - Optional API key. If not provided, will look for COHERE_API_KEY env var
    pub fn new(api_key: Option<String>) -> Result<Self> {
        let api_key = api_key
            .or_else(|| env::var("COHERE_API_KEY").ok())
            .ok_or_else(|| AiError::MissingConfiguration {
                field: "api_key".to_string(),
                description: "Cohere API key not provided. Set COHERE_API_KEY environment variable or pass it explicitly".to_string(),
            })?;

        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    fn convert_role(&self, role: &Role) -> String {
        match role {
            Role::System => "SYSTEM".to_string(),
            Role::User => "USER".to_string(),
            Role::Assistant => "CHATBOT".to_string(),
            Role::Tool => "TOOL".to_string(),
        }
    }

    fn convert_message(&self, message: &Message) -> CohereChatMessage {
        let message_text = match &message.content {
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

        CohereChatMessage {
            role: self.convert_role(&message.role),
            message: message_text,
        }
    }

    fn convert_to_standard_response(&self, response: CohereChatResponse) -> CompletionResponse {
        CompletionResponse {
            id: response.response_id.unwrap_or_else(|| "cohere_response".to_string()),
            model: response.generation_info.map(|info| info.model).unwrap_or_else(|| "command".to_string()),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::text(response.text),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(response.finish_reason.unwrap_or_else(|| "stop".to_string())),
            }],
            usage: response.meta.map(|meta| Usage {
                prompt_tokens: meta.billed_units.input_tokens.unwrap_or(0) as u32,
                completion_tokens: meta.billed_units.output_tokens.unwrap_or(0) as u32,
                total_tokens: (meta.billed_units.input_tokens.unwrap_or(0) + 
                              meta.billed_units.output_tokens.unwrap_or(0)) as u32,
            }),
        }
    }
}

#[async_trait]
impl CompletionProvider for CohereProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = "https://api.cohere.ai/v1/chat";
        
        // Extract system message as preamble
        let (preamble, chat_history) = {
            let mut preamble = None;
            let mut history = Vec::new();
            
            for msg in &request.messages {
                match msg.role {
                    Role::System => {
                        preamble = Some(match &msg.content {
                            MessageContent::Text(text) => text.clone(),
                            MessageContent::Parts(_) => continue,
                        });
                    }
                    _ => history.push(self.convert_message(msg)),
                }
            }
            
            (preamble, history)
        };

        // The last message should be from the user
        let message = chat_history.last()
            .filter(|m| m.role == "USER")
            .map(|m| m.message.clone())
            .ok_or_else(|| AiError::InvalidRequest {
                message: "Last message must be from user".to_string(),
                field: Some("messages".to_string()),
                code: None,
            })?;

        // Remove the last message from history
        let chat_history = chat_history[..chat_history.len()-1].to_vec();

        let cohere_request = CohereChatRequest {
            message,
            model: Some(request.model.clone()),
            preamble,
            chat_history: if chat_history.is_empty() { None } else { Some(chat_history) },
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            p: request.top_p,
            stop_sequences: request.stop.clone(),
            stream: false,
        };

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&cohere_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "cohere".to_string(),
                message: format!("Cohere API error: {}", error_text),
                error_code: None,
                retryable: status.is_server_error(),
            });
        }

        let cohere_response: CohereChatResponse = response.json().await?;
        Ok(self.convert_to_standard_response(cohere_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = "https://api.cohere.ai/v1/chat";
        
        // Extract system message as preamble
        let (preamble, chat_history) = {
            let mut preamble = None;
            let mut history = Vec::new();
            
            for msg in &request.messages {
                match msg.role {
                    Role::System => {
                        preamble = Some(match &msg.content {
                            MessageContent::Text(text) => text.clone(),
                            MessageContent::Parts(_) => continue,
                        });
                    }
                    _ => history.push(self.convert_message(msg)),
                }
            }
            
            (preamble, history)
        };

        // The last message should be from the user
        let message = chat_history.last()
            .filter(|m| m.role == "USER")
            .map(|m| m.message.clone())
            .ok_or_else(|| AiError::InvalidRequest {
                message: "Last message must be from user".to_string(),
                field: Some("messages".to_string()),
                code: None,
            })?;

        // Remove the last message from history
        let chat_history = chat_history[..chat_history.len()-1].to_vec();

        let cohere_request = CohereChatRequest {
            message,
            model: Some(request.model.clone()),
            preamble,
            chat_history: if chat_history.is_empty() { None } else { Some(chat_history) },
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            p: request.top_p,
            stop_sequences: request.stop.clone(),
            stream: true,
        };

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&cohere_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "cohere".to_string(),
                message: format!("Cohere API error: {}", error_text),
                error_code: None,
                retryable: status.is_server_error(),
            });
        }

        // Convert the response stream
        let stream = response.bytes_stream();
        let mapped_stream = stream.map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // Parse the server-sent event
                    let text = String::from_utf8_lossy(&chunk);
                    
                    // Cohere uses server-sent events format
                    if let Some(json_str) = text.strip_prefix("data: ") {
                        match serde_json::from_str::<CohereStreamEvent>(json_str.trim()) {
                            Ok(event) => {
                                match event.event_type.as_str() {
                                    "text-generation" => {
                                        Ok(StreamChunk {
                                            id: "cohere_stream".to_string(),
                                            choices: vec![crate::StreamChoice {
                                                index: 0,
                                                delta: crate::Delta {
                                                    role: None,
                                                    content: event.text,
                                                    tool_calls: None,
                                                },
                                                finish_reason: None,
                                            }],
                                            model: None,
                                        })
                                    }
                                    "stream-end" => {
                                        Ok(StreamChunk {
                                            id: "cohere_stream".to_string(),
                                            choices: vec![crate::StreamChoice {
                                                index: 0,
                                                delta: crate::Delta {
                                                    role: None,
                                                    content: None,
                                                    tool_calls: None,
                                                },
                                                finish_reason: Some("stop".to_string()),
                                            }],
                                            model: None,
                                        })
                                    }
                                    _ => {
                                        // Ignore other event types
                                        Ok(StreamChunk {
                                            id: "cohere_stream".to_string(),
                                            choices: vec![],
                                            model: None,
                                        })
                                    }
                                }
                            }
                            Err(e) => Err(AiError::StreamError {
                                message: format!("Failed to parse Cohere stream event: {}", e),
                                retryable: false,
                            }),
                        }
                    } else {
                        // Skip non-data lines (like empty lines)
                        Ok(StreamChunk {
                            id: "cohere_stream".to_string(),
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
        "cohere"
    }

    fn default_model(&self) -> &'static str {
        "command-r-plus"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            "command-r-plus",
            "command-r",
            "command",
            "command-light",
            "command-nightly",
        ]
    }
}

// Cohere API types

#[derive(Debug, Clone, Serialize)]
struct CohereChatRequest {
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    preamble: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_history: Option<Vec<CohereChatMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereChatMessage {
    role: String,
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
struct CohereChatResponse {
    text: String,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    generation_info: Option<GenerationInfo>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    meta: Option<ResponseMeta>,
}

#[derive(Debug, Clone, Deserialize)]
struct GenerationInfo {
    model: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponseMeta {
    billed_units: BilledUnits,
}

#[derive(Debug, Clone, Deserialize)]
struct BilledUnits {
    #[serde(default)]
    input_tokens: Option<usize>,
    #[serde(default)]
    output_tokens: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct CohereStreamEvent {
    event_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohere_provider_creation() {
        // This will fail without an API key, which is expected
        let result = CohereProvider::new(Some("test-key".to_string()));
        assert!(result.is_ok());
        
        let provider = result.unwrap();
        assert_eq!(provider.name(), "cohere");
        assert_eq!(provider.default_model(), "command-r-plus");
    }

    #[test]
    fn test_role_conversion() {
        let provider = CohereProvider::new(Some("test-key".to_string())).unwrap();
        assert_eq!(provider.convert_role(&Role::System), "SYSTEM");
        assert_eq!(provider.convert_role(&Role::User), "USER");
        assert_eq!(provider.convert_role(&Role::Assistant), "CHATBOT");
        assert_eq!(provider.convert_role(&Role::Tool), "TOOL");
    }
}