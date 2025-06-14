use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::{
    AiError, Choice, CompletionProvider, CompletionRequest, CompletionResponse, ContentPart, Delta,
    Message, MessageContent, Result, Role, StreamChunk, Usage,
};

pub struct GeminiProvider {
    client: Client,
    api_key: String,
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
    role: String,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    top_p: Option<f32>,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
    finish_reason: Option<String>,
    index: u32,
}

#[derive(Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
    #[allow(dead_code)]
    role: String,
}

#[derive(Deserialize)]
struct GeminiResponsePart {
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsage {
    prompt_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

#[async_trait]
impl CompletionProvider for GeminiProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let contents = convert_messages_to_gemini(request.messages);

        let gemini_request = GeminiRequest {
            contents,
            generation_config: Some(GenerationConfig {
                temperature: request.temperature,
                max_output_tokens: request.max_tokens,
                top_p: request.top_p,
            }),
        };

        let model_name = if request.model.starts_with("models/") {
            request.model
        } else {
            format!("models/{}", request.model)
        };

        let response = self
            .client
            .post(format!(
                "https://generativelanguage.googleapis.com/v1/{}:generateContent?key={}",
                model_name, self.api_key
            ))
            .json(&gemini_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "gemini".to_string(),
                message: format!("Gemini API error: {}", error_text),
                error_code: None,
                retryable: true,
            });
        }

        let gemini_response: GeminiResponse = response.json().await?;

        let choices = gemini_response
            .candidates
            .into_iter()
            .map(|candidate| Choice {
                index: candidate.index,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::text(
                        candidate
                            .content
                            .parts
                            .iter()
                            .map(|p| p.text.clone())
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: candidate.finish_reason,
            })
            .collect();

        let usage = gemini_response.usage_metadata.map(|u| Usage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        });

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model_name,
            choices,
            usage,
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let contents = convert_messages_to_gemini(request.messages);

        let gemini_request = GeminiRequest {
            contents,
            generation_config: Some(GenerationConfig {
                temperature: request.temperature,
                max_output_tokens: request.max_tokens,
                top_p: request.top_p,
            }),
        };

        let model_name = if request.model.starts_with("models/") {
            request.model
        } else {
            format!("models/{}", request.model)
        };

        let response = self
            .client
            .post(format!(
                "https://generativelanguage.googleapis.com/v1/{}:streamGenerateContent?key={}",
                model_name, self.api_key
            ))
            .json(&gemini_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError {
                provider: "gemini".to_string(),
                message: format!("Gemini API error: {}", error_text),
                error_code: None,
                retryable: true,
            });
        }

        let stream = response.bytes_stream();
        let stream = stream
            .map(move |result| match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    parse_gemini_stream(&text, &model_name)
                }
                Err(e) => Err(AiError::StreamError {
                    message: e.to_string(),
                    retryable: true,
                }),
            })
            .filter_map(|result| async move {
                match result {
                    Ok(Some(chunk)) => Some(Ok(chunk)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(Box::pin(stream))
    }

    fn name(&self) -> &'static str {
        "Gemini"
    }

    fn default_model(&self) -> &'static str {
        "gemini-2.0-flash-exp"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            "gemini-2.0-flash-exp",
            "gemini-exp-1206",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
    }
}

fn convert_messages_to_gemini(messages: Vec<Message>) -> Vec<GeminiContent> {
    let mut contents = Vec::new();
    let mut system_message = None;

    for message in messages {
        match message.role {
            Role::System => {
                system_message = Some(extract_text_from_content(&message.content));
            }
            Role::User => {
                let mut content = extract_text_from_content(&message.content);
                if let Some(sys) = &system_message {
                    content = format!("{}\n\n{}", sys, content);
                    system_message = None;
                }
                contents.push(GeminiContent {
                    parts: vec![GeminiPart { text: content }],
                    role: "user".to_string(),
                });
            }
            Role::Assistant => {
                contents.push(GeminiContent {
                    parts: vec![GeminiPart {
                        text: extract_text_from_content(&message.content),
                    }],
                    role: "model".to_string(),
                });
            }
            Role::Tool => {
                // Tool responses are sent as user messages in Gemini
                contents.push(GeminiContent {
                    parts: vec![GeminiPart {
                        text: extract_text_from_content(&message.content),
                    }],
                    role: "user".to_string(),
                });
            }
        }
    }

    contents
}

fn extract_text_from_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn parse_gemini_stream(data: &str, model: &str) -> Result<Option<StreamChunk>> {
    if let Ok(response) = serde_json::from_str::<GeminiResponse>(data) {
        if let Some(candidate) = response.candidates.first() {
            if let Some(part) = candidate.content.parts.first() {
                return Ok(Some(StreamChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    choices: vec![crate::StreamChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(part.text.clone()),
                            tool_calls: None,
                        },
                        finish_reason: candidate.finish_reason.clone(),
                    }],
                    model: Some(model.to_string()),
                }));
            }
        }
    }
    Ok(None)
}
