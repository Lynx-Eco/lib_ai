mod common;

use futures::StreamExt;
use lib_ai::{providers::OpenRouterProvider, CompletionProvider, MessageContent};

fn get_provider() -> Option<OpenRouterProvider> {
    match std::env::var("OPENROUTER_API_KEY") {
        Ok(key) => Some(OpenRouterProvider::new(key)),
        Err(_) => {
            eprintln!("Skipping OpenRouter tests: OPENROUTER_API_KEY not set");
            None
        }
    }
}

#[tokio::test]
async fn test_openrouter_list_models() {
    let Some(provider) = get_provider() else {
        return;
    };

    match provider.list_available_models().await {
        Ok(models) => {
            assert!(!models.is_empty());
            println!("Found {} models", models.len());

            // Print first 5 models
            for (i, model) in models.iter().take(5).enumerate() {
                println!("Model {}: {} ({})", i + 1, model.id, model.name);
                println!("  Context: {} tokens", model.context_length);
                println!(
                    "  Pricing: ${}/1M prompt, ${}/1M completion",
                    model.pricing.prompt, model.pricing.completion
                );
            }
        }
        Err(e) => {
            eprintln!("Failed to list models: {}", e);
        }
    }
}

#[tokio::test]
async fn test_openrouter_claude_completion() {
    let Some(provider) = get_provider() else {
        return;
    };

    let mut request = common::create_simple_request("anthropic/claude-3-haiku".to_string());
    request.messages[1].content = MessageContent::text("Say 'Hello from Claude via OpenRouter'");

    match provider.complete(request).await {
        Ok(response) => {
            assert!(!response.id.is_empty());
            assert_eq!(response.choices.len(), 1);

            let message = &response.choices[0].message;
            if let Some(text) = message.content.as_text() {
                assert!(text.contains("Hello"));
                println!("Claude response: {}", text);
            }
        }
        Err(e) => {
            eprintln!("Claude completion failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_openrouter_gpt_completion() {
    let Some(provider) = get_provider() else {
        return;
    };

    let mut request = common::create_simple_request("openai/gpt-3.5-turbo".to_string());
    request.messages[1].content = MessageContent::text("Say 'Hello from GPT via OpenRouter'");

    match provider.complete(request).await {
        Ok(response) => {
            assert!(!response.id.is_empty());
            assert_eq!(response.choices.len(), 1);

            let message = &response.choices[0].message;
            if let Some(text) = message.content.as_text() {
                assert!(text.contains("Hello"));
                println!("GPT response: {}", text);
            }
        }
        Err(e) => {
            eprintln!("GPT completion failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_openrouter_streaming() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_streaming_request("openai/gpt-3.5-turbo".to_string());

    match provider.complete_stream(request).await {
        Ok(mut stream) => {
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
        Err(e) => {
            eprintln!("Failed to create stream: {}", e);
        }
    }
}

#[tokio::test]
async fn test_openrouter_free_model() {
    let Some(provider) = get_provider() else {
        return;
    };

    // Test with a free model
    let mut request = common::create_simple_request("google/gemini-2.0-flash-exp:free".to_string());
    request.messages[1].content = MessageContent::text("Say 'Hello from free model'");

    match provider.complete(request).await {
        Ok(response) => {
            assert!(!response.id.is_empty());
            let text = response.choices[0].message.content.as_text().unwrap();
            assert!(text.to_lowercase().contains("hello"));
            println!("Free model response: {}", text);
        }
        Err(e) => {
            eprintln!("Free model failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_openrouter_conversation() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_conversation_request("openai/gpt-3.5-turbo".to_string());

    match provider.complete(request).await {
        Ok(response) => {
            let message = &response.choices[0].message;
            if let Some(text) = message.content.as_text() {
                assert!(text.contains("6"));
            }
        }
        Err(e) => {
            eprintln!("Conversation test failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_openrouter_json_mode() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_json_request("openai/gpt-3.5-turbo".to_string());

    match provider.complete(request).await {
        Ok(response) => {
            let message = &response.choices[0].message;
            if let Some(text) = message.content.as_text() {
                match serde_json::from_str::<serde_json::Value>(text) {
                    Ok(json) => {
                        assert!(json.is_object());
                        println!("Valid JSON: {}", json);
                    }
                    Err(e) => {
                        println!("JSON parse error: {}", e);
                        println!("Response: {}", text);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("JSON mode test failed: {}", e);
        }
    }
}
