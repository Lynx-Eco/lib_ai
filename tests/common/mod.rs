use lib_ai::{
    CompletionRequest, ContentPart, ImageUrl, Message, MessageContent, ResponseFormat,
    ResponseFormatType, Role, Tool, ToolChoice, ToolFunction, ToolType,
};
use serde_json::json;

pub fn create_simple_request(model: String) -> CompletionRequest {
    CompletionRequest {
        model,
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful assistant."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("Say 'Hello, World!' and nothing else."),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.0),
        max_tokens: Some(20),
        stream: Some(false),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        json_schema: None,
    }
}

pub fn create_streaming_request(model: String) -> CompletionRequest {
    let mut request = create_simple_request(model);
    request.stream = Some(true);
    request
}

pub fn create_tool_request(model: String) -> CompletionRequest {
    let weather_tool = Tool {
        r#type: ToolType::Function,
        function: ToolFunction {
            name: "get_weather".to_string(),
            description: Some("Get the current weather in a given location".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }),
        },
    };

    CompletionRequest {
        model,
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::text("What's the weather like in San Francisco?"),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: Some(0.0),
        max_tokens: Some(150),
        stream: Some(false),
        tools: Some(vec![weather_tool]),
        tool_choice: Some(ToolChoice::String("auto".to_string())),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        response_format: None,
        json_schema: None,
    }
}

pub fn create_json_request(model: String) -> CompletionRequest {
    CompletionRequest {
        model,
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful assistant that outputs JSON."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text(
                    "Return a JSON object with a single field 'message' containing 'Hello, World!'",
                ),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.0),
        max_tokens: Some(50),
        stream: Some(false),
        response_format: Some(ResponseFormat {
            r#type: ResponseFormatType::JsonObject,
        }),
        tools: None,
        tool_choice: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        json_schema: None,
    }
}

pub fn create_multimodal_request(model: String) -> CompletionRequest {
    CompletionRequest {
        model,
        messages: vec![
            Message {
                role: Role::User,
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Describe this image:".to_string(),
                    },
                    ContentPart::Image {
                        image_url: ImageUrl {
                            // Using a base64 encoded 1x1 red pixel for testing
                            url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.0),
        max_tokens: Some(100),
        stream: Some(false),
        tools: None,
        tool_choice: None,
        response_format: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        json_schema: None,
    }
}

pub fn create_conversation_request(model: String) -> CompletionRequest {
    CompletionRequest {
        model,
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful math tutor."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("What is 2+2?"),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::text("2+2 equals 4."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("And what is 3+3?"),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.0),
        max_tokens: Some(50),
        stream: Some(false),
        tools: None,
        tool_choice: None,
        response_format: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        json_schema: None,
    }
}
