mod common;

use lib_ai::{providers::GeminiProvider, CompletionProvider, MessageContent, Role, Message};
use futures::StreamExt;

fn get_provider() -> Option<GeminiProvider> {
    match std::env::var("GEMINI_API_KEY") {
        Ok(key) => Some(GeminiProvider::new(key)),
        Err(_) => {
            eprintln!("Skipping Gemini tests: GEMINI_API_KEY not set");
            None
        }
    }
}

#[tokio::test]
async fn test_gemini_simple_completion() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_simple_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert!(!response.id.is_empty());
    assert!(!response.choices.is_empty());
    
    let message = &response.choices[0].message;
    assert!(matches!(message.content, MessageContent::Text(_)));
    
    if let Some(text) = message.content.as_text() {
        assert!(text.to_lowercase().contains("hello"));
    }
    
    assert!(response.usage.is_some());
}

#[tokio::test]
async fn test_gemini_streaming() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_streaming_request(provider.default_model().to_string());
    let mut stream = provider.complete_stream(request).await.unwrap();
    
    let mut chunks_received = 0;
    let mut full_content = String::new();
    
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                chunks_received += 1;
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content {
                        full_content.push_str(&content);
                    }
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }
    
    assert!(chunks_received > 0);
    assert!(full_content.to_lowercase().contains("hello"));
}

#[tokio::test]
async fn test_gemini_system_message_handling() {
    let Some(provider) = get_provider() else { return };
    
    // Gemini merges system messages with the first user message
    let request = lib_ai::CompletionRequest {
        model: provider.default_model().to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You must always respond with 'SYSTEM MESSAGE WORKS'"),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("Say anything"),
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
    let text = response.choices[0].message.content.as_text().unwrap();
    assert!(text.contains("SYSTEM MESSAGE WORKS"));
}

#[tokio::test]
async fn test_gemini_conversation() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_conversation_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert!(!response.choices.is_empty());
    
    let message = &response.choices[0].message;
    if let Some(text) = message.content.as_text() {
        assert!(text.contains("6"));
    }
}

#[tokio::test]
async fn test_gemini_different_models() {
    let Some(provider) = get_provider() else { return };
    
    let models = vec![
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ];
    
    for model in models {
        println!("Testing model: {}", model);
        let mut request = common::create_simple_request(model.to_string());
        request.messages[1].content = MessageContent::text("Say 'Hello from Gemini'");
        
        match provider.complete(request).await {
            Ok(response) => {
                assert!(response.choices[0].message.content.as_text().unwrap().to_lowercase().contains("hello"));
                println!("Model {} succeeded", model);
            }
            Err(e) => {
                eprintln!("Model {} failed: {}", model, e);
            }
        }
    }
}

#[tokio::test]
async fn test_gemini_temperature() {
    let Some(provider) = get_provider() else { return };
    
    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.temperature = Some(0.0); // Deterministic
    request.messages[1].content = MessageContent::text("Complete this sequence: 1, 2, 3,");
    
    let response1 = provider.complete(request.clone()).await.unwrap();
    let response2 = provider.complete(request).await.unwrap();
    
    let text1 = response1.choices[0].message.content.as_text().unwrap();
    let text2 = response2.choices[0].message.content.as_text().unwrap();
    
    // Both should mention 4
    assert!(text1.contains("4"));
    assert!(text2.contains("4"));
}

#[tokio::test]
async fn test_gemini_max_tokens() {
    let Some(provider) = get_provider() else { return };
    
    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.max_tokens = Some(10); // Very low limit
    request.messages[1].content = MessageContent::text("Count from 1 to 100");
    
    let response = provider.complete(request).await.unwrap();
    
    // Response should be truncated
    let text = response.choices[0].message.content.as_text().unwrap();
    assert!(text.len() < 200); // Should be much shorter than counting to 100
}

#[tokio::test]
async fn test_gemini_error_handling() {
    let Some(provider) = get_provider() else { return };
    
    // Test with invalid model name
    let mut request = common::create_simple_request("invalid-model-name".to_string());
    request.messages[1].content = MessageContent::text("Hello");
    
    match provider.complete(request).await {
        Ok(_) => panic!("Expected error for invalid model"),
        Err(e) => {
            println!("Got expected error: {}", e);
            assert!(e.to_string().contains("error") || e.to_string().contains("Error"));
        }
    }
}