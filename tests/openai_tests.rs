mod common;

use futures::StreamExt;
use lib_ai::{providers::OpenAIProvider, CompletionProvider, MessageContent};

fn get_provider() -> Option<OpenAIProvider> {
    match std::env::var("OPENAI_API_KEY") {
        Ok(key) => Some(OpenAIProvider::new(key)),
        Err(_) => {
            eprintln!("Skipping OpenAI tests: OPENAI_API_KEY not set");
            None
        }
    }
}

#[tokio::test]
async fn test_openai_simple_completion() {
    let Some(provider) = get_provider() else {
        return;
    };

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
async fn test_openai_streaming() {
    let Some(provider) = get_provider() else {
        return;
    };

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
async fn test_openai_tool_calling() {
    let Some(provider) = get_provider() else {
        return;
    };

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
async fn test_openai_json_mode() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_json_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();

    assert_eq!(response.choices.len(), 1);

    let message = &response.choices[0].message;
    if let Some(text) = message.content.as_text() {
        // Verify it's valid JSON
        let json_result: Result<serde_json::Value, _> = serde_json::from_str(text);
        assert!(json_result.is_ok(), "Response should be valid JSON");

        let json = json_result.unwrap();
        assert!(json.is_object());
    }
}

#[tokio::test]
async fn test_openai_multimodal() {
    let Some(provider) = get_provider() else {
        return;
    };

    // Only test with vision-capable models
    let request = common::create_multimodal_request("gpt-4o".to_string());
    let response = provider.complete(request).await.unwrap();

    assert_eq!(response.choices.len(), 1);

    let message = &response.choices[0].message;
    assert!(message.content.as_text().is_some());
}

#[tokio::test]
async fn test_openai_conversation() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_conversation_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();

    assert_eq!(response.choices.len(), 1);

    let message = &response.choices[0].message;
    if let Some(text) = message.content.as_text() {
        // Should mention 6 as the answer to 3+3
        assert!(text.contains("6"));
    }
}

#[tokio::test]
async fn test_openai_temperature() {
    let Some(provider) = get_provider() else {
        return;
    };

    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.temperature = Some(0.0); // Deterministic
    request.messages[1].content = MessageContent::text("Generate a random number between 1 and 10");

    let response1 = provider.complete(request.clone()).await.unwrap();
    let response2 = provider.complete(request).await.unwrap();

    let text1 = response1.choices[0].message.content.as_text().unwrap();
    let text2 = response2.choices[0].message.content.as_text().unwrap();

    // With temperature 0, responses should be very similar (though not guaranteed identical)
    println!("Response 1: {}", text1);
    println!("Response 2: {}", text2);
}

#[tokio::test]
async fn test_openai_max_tokens() {
    let Some(provider) = get_provider() else {
        return;
    };

    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.max_tokens = Some(5); // Very low limit
    request.messages[1].content = MessageContent::text("Tell me a long story about dragons");

    let response = provider.complete(request).await.unwrap();

    // Check that the response was truncated
    assert_eq!(response.choices[0].finish_reason.as_deref(), Some("length"));
}
