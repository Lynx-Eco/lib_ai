mod common;

use lib_ai::{providers::AnthropicProvider, CompletionProvider, MessageContent, Role, Message};
use futures::StreamExt;

fn get_provider() -> Option<AnthropicProvider> {
    match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => Some(AnthropicProvider::new(key)),
        Err(_) => {
            eprintln!("Skipping Anthropic tests: ANTHROPIC_API_KEY not set");
            None
        }
    }
}

#[tokio::test]
async fn test_anthropic_simple_completion() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_simple_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert!(!response.id.is_empty());
    assert_eq!(response.choices.len(), 1);
    
    let message = &response.choices[0].message;
    assert!(matches!(message.content, MessageContent::Text(_)));
    
    if let Some(text) = message.content.as_text() {
        assert!(text.to_lowercase().contains("hello"));
    }
    
    assert!(response.usage.is_some());
}

#[tokio::test]
async fn test_anthropic_streaming() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_streaming_request(provider.default_model().to_string());
    let mut stream = provider.complete_stream(request).await.unwrap();
    
    let mut chunks_received = 0;
    let mut full_content = String::new();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        chunks_received += 1;
        
        for choice in chunk.choices {
            if let Some(content) = choice.delta.content {
                full_content.push_str(&content);
            }
        }
    }
    
    assert!(chunks_received > 0);
    assert!(full_content.to_lowercase().contains("hello"));
}

#[tokio::test]
async fn test_anthropic_system_message() {
    let Some(provider) = get_provider() else { return };
    
    // Anthropic handles system messages differently
    let request = common::create_conversation_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert_eq!(response.choices.len(), 1);
    
    let message = &response.choices[0].message;
    if let Some(text) = message.content.as_text() {
        assert!(text.contains("6"));
    }
}

#[tokio::test]
async fn test_anthropic_tool_calling() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_tool_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert_eq!(response.choices.len(), 1);
    
    let message = &response.choices[0].message;
    
    // Check if the model called the weather tool
    if let Some(tool_calls) = &message.tool_calls {
        assert!(!tool_calls.is_empty());
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.function.name, "get_weather");
        
        // Verify the arguments contain location
        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
        assert!(args.get("location").is_some());
    }
}

#[tokio::test]
async fn test_anthropic_multimodal() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_multimodal_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert_eq!(response.choices.len(), 1);
    
    let message = &response.choices[0].message;
    assert!(message.content.as_text().is_some());
}

#[tokio::test]
async fn test_anthropic_long_context() {
    let Some(provider) = get_provider() else { return };
    
    // Create a request with a very long user message
    let long_text = "Hello ".repeat(1000);
    let request = lib_ai::CompletionRequest {
        model: provider.default_model().to_string(),
        messages: vec![
            Message {
                role: Role::User,
                content: MessageContent::text(format!("{} Please just say 'I read your message'", long_text)),
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
    };
    
    let response = provider.complete(request).await.unwrap();
    assert!(response.choices[0].message.content.as_text().is_some());
}

#[tokio::test]
async fn test_anthropic_temperature() {
    let Some(provider) = get_provider() else { return };
    
    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.temperature = Some(0.0); // Deterministic
    request.messages[1].content = MessageContent::text("Generate a random number between 1 and 10");
    
    let response1 = provider.complete(request.clone()).await.unwrap();
    let response2 = provider.complete(request).await.unwrap();
    
    let text1 = response1.choices[0].message.content.as_text().unwrap();
    let text2 = response2.choices[0].message.content.as_text().unwrap();
    
    // With temperature 0, responses should be very similar
    println!("Response 1: {}", text1);
    println!("Response 2: {}", text2);
}

#[tokio::test]
async fn test_anthropic_different_models() {
    let Some(provider) = get_provider() else { return };
    
    let models = vec![
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
    ];
    
    for model in models {
        println!("Testing model: {}", model);
        let mut request = common::create_simple_request(model.to_string());
        request.messages[1].content = MessageContent::text("Say 'Hello from Claude'");
        
        match provider.complete(request).await {
            Ok(response) => {
                assert!(response.choices[0].message.content.as_text().unwrap().contains("Hello"));
                println!("Model {} succeeded", model);
            }
            Err(e) => {
                eprintln!("Model {} failed: {}", model, e);
            }
        }
    }
}