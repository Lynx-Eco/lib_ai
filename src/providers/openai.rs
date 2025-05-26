use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;

use crate::{
    CompletionProvider, CompletionRequest, CompletionResponse, StreamChunk, Result, AiError, 
    Role, MessageContent, ToolCall, Tool, ToolChoice, ResponseFormat, ContentPart, Message,
    Choice, Usage, Delta, StreamChoice, ToolCallDelta
};

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, "https://api.openai.com/v1".to_string())
    }

    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
        }
    }

    fn convert_message(&self, msg: Message) -> OpenAIMessage {
        let content = match msg.content {
            MessageContent::Text(text) => OpenAIContent::String(text),
            MessageContent::Parts(parts) => OpenAIContent::Array(
                parts.into_iter().map(|part| match part {
                    ContentPart::Text { text } => OpenAIContentPart {
                        r#type: "text".to_string(),
                        text: Some(text),
                        image_url: None,
                    },
                    ContentPart::Image { image_url } => OpenAIContentPart {
                        r#type: "image_url".to_string(),
                        text: None,
                        image_url: Some(image_url),
                    },
                }).collect()
            ),
        };

        OpenAIMessage {
            role: match msg.role {
                Role::System => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
                Role::Tool => "tool".to_string(),
            },
            content: Some(content),
            tool_calls: msg.tool_calls,
            tool_call_id: msg.tool_call_id,
        }
    }

    fn convert_response(&self, resp: OpenAIResponse) -> CompletionResponse {
        CompletionResponse {
            id: resp.id,
            model: resp.model,
            choices: resp.choices.into_iter().map(|c| Choice {
                index: c.index,
                message: Message {
                    role: match c.message.role.as_str() {
                        "system" => Role::System,
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        _ => Role::Assistant,
                    },
                    content: match c.message.content {
                        Some(OpenAIContent::String(s)) => MessageContent::Text(s),
                        Some(OpenAIContent::Array(parts)) => MessageContent::Parts(
                            parts.into_iter().filter_map(|p| {
                                if p.r#type == "text" {
                                    p.text.map(|text| ContentPart::Text { text })
                                } else if p.r#type == "image_url" {
                                    p.image_url.map(|image_url| ContentPart::Image { image_url })
                                } else {
                                    None
                                }
                            }).collect()
                        ),
                        None => MessageContent::Text("".to_string()),
                    },
                    tool_calls: c.message.tool_calls,
                    tool_call_id: None,
                },
                finish_reason: c.finish_reason,
            }).collect(),
            usage: resp.usage,
        }
    }
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum OpenAIContent {
    String(String),
    Array(Vec<OpenAIContentPart>),
}

#[derive(Serialize, Deserialize)]
struct OpenAIContentPart {
    r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<crate::ImageUrl>,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    index: u32,
    message: OpenAIMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIStreamChunk {
    id: String,
    #[allow(dead_code)]
    object: String,
    #[allow(dead_code)]
    created: u64,
    model: String,
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAIStreamChoice {
    index: u32,
    delta: OpenAIDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[async_trait]
impl CompletionProvider for OpenAIProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let openai_request = OpenAIRequest {
            model: request.model,
            messages: request.messages.into_iter().map(|m| self.convert_message(m)).collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(false),
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop,
            tools: request.tools,
            tool_choice: request.tool_choice,
            response_format: request.response_format,
        };

        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError(format!("OpenAI API error: {}", error_text)));
        }

        let openai_response: OpenAIResponse = response.json().await?;
        Ok(self.convert_response(openai_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let openai_request = OpenAIRequest {
            model: request.model,
            messages: request.messages.into_iter().map(|m| self.convert_message(m)).collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(true),
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop,
            tools: request.tools,
            tool_choice: request.tool_choice,
            response_format: request.response_format,
        };

        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError(format!("OpenAI API error: {}", error_text)));
        }

        let stream = response.bytes_stream();
        let stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    parse_openai_sse(&text)
                }
                Err(e) => Err(AiError::StreamError(e.to_string())),
            }
        }).filter_map(|result| async move {
            match result {
                Ok(Some(chunk)) => Some(Ok(chunk)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        });

        Ok(Box::pin(stream))
    }

    fn name(&self) -> &'static str {
        "OpenAI"
    }

    fn default_model(&self) -> &'static str {
        "gpt-4o"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
        ]
    }
}

fn parse_openai_sse(data: &str) -> Result<Option<StreamChunk>> {
    for line in data.lines() {
        if line.starts_with("data: ") {
            let json_str = &line[6..];
            if json_str == "[DONE]" {
                return Ok(None);
            }
            
            if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                return Ok(Some(StreamChunk {
                    id: chunk.id,
                    choices: chunk.choices.into_iter().map(|c| StreamChoice {
                        index: c.index,
                        delta: Delta {
                            role: c.delta.role.map(|r| match r.as_str() {
                                "system" => Role::System,
                                "user" => Role::User,
                                "assistant" => Role::Assistant,
                                "tool" => Role::Tool,
                                _ => Role::User,
                            }),
                            content: c.delta.content,
                            tool_calls: c.delta.tool_calls,
                        },
                        finish_reason: c.finish_reason,
                    }).collect(),
                    model: Some(chunk.model),
                }));
            }
        }
    }
    Ok(None)
}