use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::env;
use std::time::Duration;
use tokio::time::sleep;

use crate::{
    CompletionProvider, CompletionRequest, CompletionResponse, StreamChunk,
    Message, MessageContent, Role, Choice, Usage, AiError, Result,
};

/// Replicate provider for open-source models
pub struct ReplicateProvider {
    client: Client,
    api_key: String,
}

impl ReplicateProvider {
    /// Create a new Replicate provider
    /// 
    /// # Arguments
    /// * `api_key` - Optional API key. If not provided, will look for REPLICATE_API_TOKEN env var
    pub fn new(api_key: Option<String>) -> Result<Self> {
        let api_key = api_key
            .or_else(|| env::var("REPLICATE_API_TOKEN").ok())
            .ok_or_else(|| AiError::MissingConfiguration {
                field: "api_key".to_string(),
                description: "Replicate API token not provided. Set REPLICATE_API_TOKEN environment variable or pass it explicitly".to_string(),
            })?;

        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    /// Get model version ID for a given model identifier
    async fn get_model_version(&self, model: &str) -> Result<String> {
        // For now, we'll use a mapping of known models to their versions
        // In a production system, you'd want to fetch this from the Replicate API
        let version = match model {
            "meta/llama-2-70b-chat" => "02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            "meta/llama-2-13b-chat" => "f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
            "meta/llama-2-7b-chat" => "13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",
            "mistralai/mistral-7b-instruct-v0.2" => "6282abe8f29b89d2b27b8a36a215b2f529459ee712ba9c5e44bdc96ca35b9cdc",
            "stability-ai/sdxl" => "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            _ => {
                // Try to use the model string as-is (might be a full version ID)
                model
            }
        }.to_string();

        Ok(version)
    }

    /// Format messages for Replicate models
    fn format_prompt(&self, messages: &[Message]) -> String {
        let mut prompt = String::new();
        
        for message in messages {
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

            match message.role {
                Role::System => {
                    prompt.push_str(&format!("System: {}\n\n", content));
                }
                Role::User => {
                    prompt.push_str(&format!("Human: {}\n\n", content));
                }
                Role::Assistant => {
                    prompt.push_str(&format!("Assistant: {}\n\n", content));
                }
                Role::Tool => {
                    prompt.push_str(&format!("Tool: {}\n\n", content));
                }
            }
        }
        
        // Add the assistant prompt to get a response
        prompt.push_str("Assistant: ");
        
        prompt
    }

    /// Wait for a prediction to complete
    async fn wait_for_prediction(&self, prediction_url: &str) -> Result<ReplicatePrediction> {
        let max_attempts = 300; // 5 minutes with 1 second intervals
        
        for _ in 0..max_attempts {
            let response = self.client
                .get(prediction_url)
                .header("Authorization", format!("Token {}", self.api_key))
                .send()
                .await?;

            if !response.status().is_success() {
                let error_text = response.text().await?;
                return Err(AiError::ProviderError {
                    provider: "replicate".to_string(),
                    message: format!("Failed to get prediction status: {}", error_text),
                    error_code: None,
                    retryable: false,
                });
            }

            let prediction: ReplicatePrediction = response.json().await?;
            
            match prediction.status.as_str() {
                "succeeded" => return Ok(prediction),
                "failed" | "canceled" => {
                    return Err(AiError::ProviderError {
                        provider: "replicate".to_string(),
                        message: format!("Prediction {}: {}", prediction.status, 
                            prediction.error.unwrap_or_else(|| "Unknown error".to_string())),
                        error_code: None,
                        retryable: false,
                    });
                }
                "starting" | "processing" => {
                    // Still running, wait and retry
                    sleep(Duration::from_secs(1)).await;
                }
                _ => {
                    return Err(AiError::ProviderError {
                        provider: "replicate".to_string(),
                        message: format!("Unknown prediction status: {}", prediction.status),
                        error_code: None,
                        retryable: false,
                    });
                }
            }
        }
        
        Err(AiError::TimeoutError {
            timeout: Duration::from_secs(300),
            retryable: false,
        })
    }
}

#[async_trait]
impl CompletionProvider for ReplicateProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = "https://api.replicate.com/v1/predictions";
        
        // Get the model version
        let version = self.get_model_version(&request.model).await?;
        
        // Format the prompt
        let prompt = self.format_prompt(&request.messages);
        
        // Build the input parameters
        let mut input = serde_json::json!({
            "prompt": prompt,
        });
        
        if let Some(temp) = request.temperature {
            input["temperature"] = serde_json::json!(temp);
        }
        
        if let Some(max_tokens) = request.max_tokens {
            input["max_new_tokens"] = serde_json::json!(max_tokens);
        }
        
        if let Some(top_p) = request.top_p {
            input["top_p"] = serde_json::json!(top_p);
        }
        
        if let Some(stop) = &request.stop {
            input["stop_sequences"] = serde_json::json!(stop.join(","));
        }

        let replicate_request = ReplicateCreatePrediction {
            version,
            input,
            webhook: None,
            webhook_events_filter: None,
        };

        // Create the prediction
        let response = self.client
            .post(url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&replicate_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "replicate".to_string(),
                message: format!("Replicate API error: {}", error_text),
                error_code: None,
                retryable: response.status().is_server_error(),
            });
        }

        let prediction: ReplicatePrediction = response.json().await?;
        
        // Wait for the prediction to complete
        let completed_prediction = self.wait_for_prediction(&prediction.urls.get).await?;
        
        // Extract the output
        let output_text = match &completed_prediction.output {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(arr)) => {
                // Some models return an array of strings
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join("")
            }
            _ => String::new(),
        };

        Ok(CompletionResponse {
            id: completed_prediction.id,
            model: request.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::text(output_text),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None, // Replicate doesn't provide token usage info
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Replicate doesn't support true streaming for language models
        // We'll simulate it by getting the full response and streaming it back
        let response = self.complete(request).await?;
        
        let text = response.choices[0].message.content.as_text()
            .unwrap_or("")
            .to_string();
        
        // Split the text into chunks
        let chunks: Vec<String> = text
            .chars()
            .collect::<Vec<_>>()
            .chunks(10) // Stream 10 characters at a time
            .map(|chunk| chunk.iter().collect::<String>())
            .collect();
        
        let stream = futures::stream::iter(chunks.into_iter().enumerate().map(|(i, chunk)| {
            Ok(StreamChunk {
                id: "replicate_stream".to_string(),
                choices: vec![crate::StreamChoice {
                    index: 0,
                    delta: crate::Delta {
                        role: if i == 0 { Some(Role::Assistant) } else { None },
                        content: Some(chunk),
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
                model: None,
            })
        }).chain(std::iter::once(Ok(StreamChunk {
            id: "replicate_stream".to_string(),
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
        }))));

        Ok(Box::pin(stream))
    }

    fn name(&self) -> &'static str {
        "replicate"
    }

    fn default_model(&self) -> &'static str {
        "meta/llama-2-70b-chat"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            "meta/llama-2-70b-chat",
            "meta/llama-2-13b-chat",
            "meta/llama-2-7b-chat",
            "mistralai/mistral-7b-instruct-v0.2",
            "stability-ai/sdxl", // For image generation
        ]
    }
}

// Replicate API types

#[derive(Debug, Clone, Serialize)]
struct ReplicateCreatePrediction {
    version: String,
    input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    webhook: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    webhook_events_filter: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReplicatePrediction {
    id: String,
    status: String,
    #[serde(default)]
    output: Option<Value>,
    #[serde(default)]
    error: Option<String>,
    urls: PredictionUrls,
}

#[derive(Debug, Clone, Deserialize)]
struct PredictionUrls {
    get: String,
    cancel: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replicate_provider_creation() {
        let result = ReplicateProvider::new(Some("test-token".to_string()));
        assert!(result.is_ok());
        
        let provider = result.unwrap();
        assert_eq!(provider.name(), "replicate");
        assert_eq!(provider.default_model(), "meta/llama-2-70b-chat");
    }

    #[test]
    fn test_prompt_formatting() {
        let provider = ReplicateProvider::new(Some("test-token".to_string())).unwrap();
        
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are helpful"),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("Hello"),
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        
        let prompt = provider.format_prompt(&messages);
        assert_eq!(prompt, "System: You are helpful\n\nHuman: Hello\n\nAssistant: ");
    }
}