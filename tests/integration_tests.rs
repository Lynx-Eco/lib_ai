mod common;

use lib_ai::{
    providers::*, CompletionProvider, MessageContent, Role, Message,
    ToolCall, ToolType, FunctionCall,
};
use std::sync::Arc;

// Test that all providers implement the same interface
#[tokio::test]
async fn test_provider_trait_consistency() {
    // This test just verifies compilation - that all providers implement CompletionProvider
    fn assert_provider<T: CompletionProvider>() {}
    
    assert_provider::<OpenAIProvider>();
    assert_provider::<AnthropicProvider>();
    assert_provider::<GeminiProvider>();
    assert_provider::<XAIProvider>();
    assert_provider::<OpenRouterProvider>();
}

// Test provider switching with dynamic dispatch
#[tokio::test]
async fn test_dynamic_provider_switching() {
    let providers: Vec<(String, Arc<dyn CompletionProvider>)> = vec![
        ("OpenAI".to_string(), Arc::new(OpenAIProvider::new("test-key".to_string()))),
        ("Anthropic".to_string(), Arc::new(AnthropicProvider::new("test-key".to_string()))),
        ("Gemini".to_string(), Arc::new(GeminiProvider::new("test-key".to_string()))),
        ("xAI".to_string(), Arc::new(XAIProvider::new("test-key".to_string()))),
        ("OpenRouter".to_string(), Arc::new(OpenRouterProvider::new("test-key".to_string()))),
    ];
    
    for (name, provider) in providers {
        assert_eq!(provider.name(), name);
        assert!(!provider.default_model().is_empty());
        assert!(!provider.available_models().is_empty());
    }
}

// Test tool result handling
#[tokio::test]
async fn test_tool_result_message() {
    // Create a tool result message
    let tool_result = Message {
        role: Role::Tool,
        content: MessageContent::text(r#"{"temperature": 72, "condition": "sunny"}"#),
        tool_calls: None,
        tool_call_id: Some("call_123".to_string()),
    };
    
    // Verify serialization
    let json = serde_json::to_string(&tool_result).unwrap();
    assert!(json.contains("tool"));
    assert!(json.contains("call_123"));
}

// Test message content variants
#[tokio::test]
async fn test_message_content_variants() {
    // Test text content
    let text_content = MessageContent::text("Hello, world!");
    assert_eq!(text_content.as_text(), Some("Hello, world!"));
    
    // Test multipart content
    let parts_content = MessageContent::Parts(vec![
        lib_ai::ContentPart::Text { text: "Check out this image:".to_string() },
        lib_ai::ContentPart::Image { 
            image_url: lib_ai::ImageUrl {
                url: "https://example.com/image.jpg".to_string(),
                detail: Some("high".to_string()),
            }
        },
    ]);
    assert!(parts_content.as_text().is_none());
    
    // Test serialization
    let text_json = serde_json::to_string(&text_content).unwrap();
    assert!(text_json.contains("Hello, world!"));
    
    let parts_json = serde_json::to_string(&parts_content).unwrap();
    assert!(parts_json.contains("text"));
    assert!(parts_json.contains("image_url"));
}

// Test tool call creation
#[tokio::test]
async fn test_tool_call_creation() {
    let tool_call = ToolCall {
        id: "call_abc123".to_string(),
        r#type: ToolType::Function,
        function: FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location": "San Francisco", "unit": "celsius"}"#.to_string(),
        },
    };
    
    // Verify serialization
    let json = serde_json::to_string(&tool_call).unwrap();
    assert!(json.contains("call_abc123"));
    assert!(json.contains("get_weather"));
    assert!(json.contains("San Francisco"));
    
    // Verify deserialization
    let parsed: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "call_abc123");
    assert_eq!(parsed.function.name, "get_weather");
}

// Test error handling consistency
#[tokio::test]
async fn test_error_handling() {
    use lib_ai::AiError;
    
    // Test error variants
    let errors = vec![
        AiError::InvalidApiKey { provider: "test".to_string() },
        AiError::RateLimitExceeded { 
            retry_after: None,
            daily_limit: None,
            requests_remaining: None,
        },
        AiError::InvalidRequest { 
            message: "Missing required field".to_string(),
            field: None,
            code: None,
        },
        AiError::ProviderError { 
            provider: "test".to_string(),
            message: "Service unavailable".to_string(),
            error_code: None,
            retryable: false,
        },
        AiError::StreamError { 
            message: "Connection lost".to_string(),
            retryable: false,
        },
    ];
    
    for error in errors {
        // All errors should have reasonable string representations
        let error_str = error.to_string();
        assert!(!error_str.is_empty());
        
        // Test debug formatting
        let debug_str = format!("{:?}", error);
        assert!(!debug_str.is_empty());
    }
}

// Test request builder pattern
#[tokio::test]
async fn test_request_builder_pattern() {
    // Start with minimal request
    let mut request = lib_ai::CompletionRequest {
        model: "test-model".to_string(),
        messages: vec![
            Message {
                role: Role::User,
                content: MessageContent::text("Hello"),
                tool_calls: None,
                tool_call_id: None,
            }
        ],
        temperature: None,
        max_tokens: None,
        stream: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        json_schema: None,
    };
    
    // Add options incrementally
    request.temperature = Some(0.7);
    request.max_tokens = Some(100);
    request.stream = Some(true);
    
    // Verify all fields are set correctly
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.max_tokens, Some(100));
    assert_eq!(request.stream, Some(true));
}

// Test model info consistency
#[tokio::test]
async fn test_model_info() {
    let providers: Vec<Box<dyn CompletionProvider>> = vec![
        Box::new(OpenAIProvider::new("test".to_string())),
        Box::new(AnthropicProvider::new("test".to_string())),
        Box::new(GeminiProvider::new("test".to_string())),
        Box::new(XAIProvider::new("test".to_string())),
        Box::new(OpenRouterProvider::new("test".to_string())),
    ];
    
    for provider in providers {
        let default_model = provider.default_model();
        let available_models = provider.available_models();
        
        // Default model should be in available models
        assert!(
            available_models.contains(&default_model),
            "{} default model {} not in available models",
            provider.name(),
            default_model
        );
    }
}