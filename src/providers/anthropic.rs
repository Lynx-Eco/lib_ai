use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;

use crate::{
    CompletionProvider, CompletionRequest, CompletionResponse, StreamChunk, Result, AiError, 
    Message, Role, Choice, Usage, Delta, StreamChoice, MessageContent, ContentPart, 
    ToolCall, ToolChoice, ToolCallDelta, ToolType, FunctionCall
};
use serde_json::Value;

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicMessageContent {
    Text(String),
    Parts(Vec<AnthropicContentPart>),
}

#[derive(Serialize, Deserialize)]
struct AnthropicContentPart {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<AnthropicImageSource>,
}

#[derive(Serialize, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    #[allow(dead_code)]
    role: String,
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<Value>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicToolChoice {
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(flatten)]
    data: serde_json::Value,
}

#[async_trait]
impl CompletionProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let (system, messages) = split_system_message(request.messages);
        
        // Convert tools if present
        let tools = request.tools.map(|tools| {
            tools.into_iter().map(|tool| AnthropicTool {
                name: tool.function.name,
                description: tool.function.description.unwrap_or_default(),
                input_schema: tool.function.parameters,
            }).collect()
        });
        
        // Convert tool choice if present
        let tool_choice = request.tool_choice.map(|tc| match tc {
            ToolChoice::String(s) => match s.as_str() {
                "auto" => AnthropicToolChoice::Auto,
                "any" => AnthropicToolChoice::Any,
                _ => AnthropicToolChoice::Auto,
            },
            ToolChoice::Object(obj) => AnthropicToolChoice::Tool {
                name: obj.function.name,
            },
        });
        
        let anthropic_request = AnthropicRequest {
            model: request.model,
            messages: messages.into_iter().map(|m| convert_message_to_anthropic(m)).collect(),
            max_tokens: request.max_tokens.unwrap_or(1024),
            temperature: request.temperature,
            stream: Some(false),
            system,
            tools,
            tool_choice,
        };

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("X-API-Key", &self.api_key)
            .header("anthropic-version", "2024-10-22")
            .header("Content-Type", "application/json")
            .json(&anthropic_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError { provider: "anthropic".to_string(), message: format!("Anthropic API error: {}", error_text), error_code: None, retryable: true });
        }

        let anthropic_response: AnthropicResponse = response.json().await?;
        
        // Extract text content and tool calls
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        
        for content in anthropic_response.content {
            match content.content_type.as_str() {
                "text" => {
                    if let Some(text) = content.text {
                        text_parts.push(text);
                    }
                },
                "tool_use" => {
                    if let (Some(id), Some(name), Some(input)) = (content.id, content.name, content.input) {
                        tool_calls.push(ToolCall {
                            id,
                            r#type: ToolType::Function,
                            function: FunctionCall {
                                name,
                                arguments: serde_json::to_string(&input).unwrap_or_default(),
                            },
                        });
                    }
                },
                _ => {},
            }
        }
        
        let message_content = if text_parts.is_empty() {
            MessageContent::Text("".to_string())
        } else {
            MessageContent::Text(text_parts.join(""))
        };
        
        Ok(CompletionResponse {
            id: anthropic_response.id,
            model: anthropic_response.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: message_content,
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: anthropic_response.usage.input_tokens,
                completion_tokens: anthropic_response.usage.output_tokens,
                total_tokens: anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
            }),
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let (system, messages) = split_system_message(request.messages);
        
        // Convert tools if present
        let tools = request.tools.map(|tools| {
            tools.into_iter().map(|tool| AnthropicTool {
                name: tool.function.name,
                description: tool.function.description.unwrap_or_default(),
                input_schema: tool.function.parameters,
            }).collect()
        });
        
        // Convert tool choice if present
        let tool_choice = request.tool_choice.map(|tc| match tc {
            ToolChoice::String(s) => match s.as_str() {
                "auto" => AnthropicToolChoice::Auto,
                "any" => AnthropicToolChoice::Any,
                _ => AnthropicToolChoice::Auto,
            },
            ToolChoice::Object(obj) => AnthropicToolChoice::Tool {
                name: obj.function.name,
            },
        });
        
        let anthropic_request = AnthropicRequest {
            model: request.model,
            messages: messages.into_iter().map(|m| convert_message_to_anthropic(m)).collect(),
            max_tokens: request.max_tokens.unwrap_or(1024),
            temperature: request.temperature,
            stream: Some(true),
            system,
            tools,
            tool_choice,
        };

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("X-API-Key", &self.api_key)
            .header("anthropic-version", "2024-10-22")
            .header("Content-Type", "application/json")
            .json(&anthropic_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AiError::ProviderError { provider: "anthropic".to_string(), message: format!("Anthropic API error: {}", error_text), error_code: None, retryable: true });
        }

        let stream = response.bytes_stream();
        let stream = stream.map(|result| {
            match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    parse_anthropic_sse(&text)
                }
                Err(e) => Err(AiError::StreamError { message: e.to_string(), retryable: true }),
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
        "Anthropic"
    }

    fn default_model(&self) -> &'static str {
        "claude-3-5-sonnet-20241022"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
    }
}

fn convert_message_to_anthropic(msg: Message) -> AnthropicMessage {
    let content = match msg.content {
        MessageContent::Text(text) => AnthropicMessageContent::Text(text),
        MessageContent::Parts(parts) => AnthropicMessageContent::Parts(
            parts.into_iter().map(|part| match part {
                ContentPart::Text { text } => AnthropicContentPart {
                    content_type: "text".to_string(),
                    text: Some(text),
                    source: None,
                },
                ContentPart::Image { image_url } => {
                    // Anthropic expects base64 images
                    if let Some(data_url) = image_url.url.strip_prefix("data:") {
                        if let Some((media_type, data)) = data_url.split_once(";base64,") {
                            AnthropicContentPart {
                                content_type: "image".to_string(),
                                text: None,
                                source: Some(AnthropicImageSource {
                                    source_type: "base64".to_string(),
                                    media_type: media_type.to_string(),
                                    data: data.to_string(),
                                }),
                            }
                        } else {
                            // Fallback to text if not base64
                            AnthropicContentPart {
                                content_type: "text".to_string(),
                                text: Some(format!("[Image: {}]", image_url.url)),
                                source: None,
                            }
                        }
                    } else {
                        // URL images not supported by Anthropic, convert to text
                        AnthropicContentPart {
                            content_type: "text".to_string(),
                            text: Some(format!("[Image: {}]", image_url.url)),
                            source: None,
                        }
                    }
                },
            }).collect()
        ),
    };
    
    AnthropicMessage {
        role: match msg.role {
            Role::User => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
            Role::System => "user".to_string(), // Anthropic doesn't have system role
            Role::Tool => "user".to_string(), // Tool results are sent as user messages
        },
        content,
    }
}

fn extract_text_from_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Parts(parts) => {
            parts.iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

fn split_system_message(messages: Vec<Message>) -> (Option<String>, Vec<Message>) {
    let mut system = None;
    let mut other_messages = Vec::new();
    
    for message in messages {
        match message.role {
            Role::System => {
                if system.is_none() {
                    system = Some(extract_text_from_content(&message.content));
                } else {
                    system = Some(format!("{}\n\n{}", system.unwrap(), extract_text_from_content(&message.content)));
                }
            }
            _ => other_messages.push(message),
        }
    }
    
    (system, other_messages)
}

fn parse_anthropic_sse(data: &str) -> Result<Option<StreamChunk>> {
    for line in data.lines() {
        if line.starts_with("event: ") {
            let event_type = &line[7..];
            
            // Find the corresponding data line
            if let Some(data_line) = data.lines().find(|l| l.starts_with("data: ")) {
                let json_str = &data_line[6..];
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                    match event_type {
                        "content_block_delta" => {
                            if let Some(delta) = json.get("delta") {
                                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                    return Ok(Some(StreamChunk {
                                        id: "stream".to_string(),
                                        choices: vec![StreamChoice {
                                            index: 0,
                                            delta: Delta {
                                                role: None,
                                                content: Some(text.to_string()),
                                                tool_calls: None,
                                            },
                                            finish_reason: None,
                                        }],
                                        model: None,
                                    }));
                                }
                            }
                        },
                        "content_block_start" => {
                            if let Some(content_block) = json.get("content_block") {
                                if content_block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                                    // Handle tool call start
                                    if let (Some(id), Some(name)) = (
                                        content_block.get("id").and_then(|i| i.as_str()),
                                        content_block.get("name").and_then(|n| n.as_str())
                                    ) {
                                        return Ok(Some(StreamChunk {
                                            id: "stream".to_string(),
                                            choices: vec![StreamChoice {
                                                index: 0,
                                                delta: Delta {
                                                    role: None,
                                                    content: None,
                                                    tool_calls: Some(vec![ToolCallDelta {
                                                        index: Some(0),
                                                        id: Some(id.to_string()),
                                                        r#type: Some(ToolType::Function),
                                                        function: Some(crate::FunctionCallDelta {
                                                            name: Some(name.to_string()),
                                                            arguments: Some("".to_string()),
                                                        }),
                                                    }]),
                                                },
                                                finish_reason: None,
                                            }],
                                            model: None,
                                        }));
                                    }
                                }
                            }
                        },
                        _ => {},
                    }
                }
            }
        }
    }
    Ok(None)
}