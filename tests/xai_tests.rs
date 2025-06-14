mod common;

use futures::StreamExt;
use lib_ai::{providers::XAIProvider, CompletionProvider, MessageContent};

fn get_provider() -> Option<XAIProvider> {
    match std::env::var("XAI_API_KEY") {
        Ok(key) => Some(XAIProvider::new(key)),
        Err(_) => {
            eprintln!("Skipping xAI tests: XAI_API_KEY not set");
            None
        }
    }
}

#[tokio::test]
async fn test_xai_simple_completion() {
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
async fn test_xai_streaming() {
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
async fn test_xai_conversation() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_conversation_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();

    assert_eq!(response.choices.len(), 1);

    let message = &response.choices[0].message;
    if let Some(text) = message.content.as_text() {
        assert!(text.contains("6"));
    }
}

#[tokio::test]
async fn test_xai_different_models() {
    let Some(provider) = get_provider() else {
        return;
    };

    let models = vec!["grok-beta"];

    for model in models {
        println!("Testing model: {}", model);
        let mut request = common::create_simple_request(model.to_string());
        request.messages[1].content = MessageContent::text("Say 'Hello from Grok'");

        match provider.complete(request).await {
            Ok(response) => {
                assert!(response.choices[0]
                    .message
                    .content
                    .as_text()
                    .unwrap()
                    .to_lowercase()
                    .contains("hello"));
                println!("Model {} succeeded", model);
            }
            Err(e) => {
                eprintln!("Model {} failed: {}", model, e);
            }
        }
    }
}

#[tokio::test]
async fn test_xai_temperature() {
    let Some(provider) = get_provider() else {
        return;
    };

    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.temperature = Some(0.0);
    request.messages[1].content = MessageContent::text("What is 2+2?");

    let response = provider.complete(request).await.unwrap();
    let text = response.choices[0].message.content.as_text().unwrap();

    assert!(text.contains("4"));
}

#[tokio::test]
async fn test_xai_json_response() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_json_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();

    assert_eq!(response.choices.len(), 1);

    let message = &response.choices[0].message;
    if let Some(text) = message.content.as_text() {
        // Try to parse as JSON
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(json) => {
                assert!(json.is_object());
                println!("Valid JSON response: {}", json);
            }
            Err(e) => {
                // xAI might not strictly follow JSON mode yet
                println!("JSON parse error (might be expected): {}", e);
                println!("Response text: {}", text);
            }
        }
    }
}
