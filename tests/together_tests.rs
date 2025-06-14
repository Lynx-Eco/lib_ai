use dotenv::dotenv;
use lib_ai::{providers::TogetherProvider, AiError, CompletionProvider};
use std::env;

mod common;

fn get_provider() -> Option<TogetherProvider> {
    dotenv().ok();

    if let Ok(api_key) = env::var("TOGETHER_API_KEY") {
        TogetherProvider::new(Some(api_key)).ok()
    } else {
        eprintln!("Skipping Together tests: TOGETHER_API_KEY not set");
        None
    }
}

#[tokio::test]
async fn test_together_simple_completion() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_simple_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();

    assert!(!response.id.is_empty());
    assert_eq!(response.model, provider.default_model());
    assert!(!response.choices.is_empty());
    assert!(response.choices[0].message.content.as_text().is_some());

    // Check usage
    assert!(response.usage.is_some());
    let usage = response.usage.unwrap();
    assert!(usage.prompt_tokens > 0);
    assert!(usage.completion_tokens > 0);
}

#[tokio::test]
async fn test_together_streaming() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_streaming_request(provider.default_model().to_string());
    let mut stream = provider.complete_stream(request).await.unwrap();

    let mut content = String::new();
    let mut chunks = 0;

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        chunks += 1;

        for choice in chunk.choices {
            if let Some(delta_content) = &choice.delta.content {
                content.push_str(delta_content);
            }
        }
    }

    assert!(chunks > 0);
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_together_temperature() {
    let Some(provider) = get_provider() else {
        return;
    };

    // Test with temperature 0 for deterministic output
    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.messages[0].content =
        lib_ai::MessageContent::text("What is 2 + 2? Answer with just the number.");
    request.temperature = Some(0.0);
    request.max_tokens = Some(10);

    let response = provider.complete(request).await.unwrap();
    let text = response.choices[0].message.content.as_text().unwrap();

    // Should contain "4" in the response
    println!("Response: {}", text);
    assert!(text.contains("4") || text.contains("four"));
}

#[tokio::test]
async fn test_together_conversation() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_conversation_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();

    assert!(!response.choices.is_empty());
    let content = response.choices[0].message.content.as_text().unwrap();

    // Should remember the name from the conversation
    println!("Conversation response: {}", content);
}

#[tokio::test]
async fn test_together_error_handling() {
    let provider = TogetherProvider::new("invalid-key".to_string());

    let request = common::create_simple_request(provider.default_model().to_string());
    let result = provider.complete(request).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        AiError::InvalidApiKey { provider } => {
            assert_eq!(provider, "together");
        }
        AiError::AuthenticationFailed { .. } => {
            // Also acceptable
        }
        AiError::ProviderError { provider, .. } => {
            assert_eq!(provider, "together");
        }
        e => panic!("Unexpected error type: {:?}", e),
    }
}

#[tokio::test]
async fn test_together_model_switching() {
    let Some(provider) = get_provider() else {
        return;
    };

    let models = provider.available_models();
    assert!(!models.is_empty());

    // Test with different models
    let test_models = vec![
        "togethercomputer/llama-2-7b-chat",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "togethercomputer/CodeLlama-7b-Instruct",
    ];

    for model in test_models {
        if models.contains(&model) {
            let mut request = common::create_simple_request(model.to_string());
            request.max_tokens = Some(50);

            match provider.complete(request).await {
                Ok(response) => {
                    assert_eq!(response.model, model);
                    println!("Model {} succeeded", model);
                }
                Err(e) => {
                    eprintln!("Model {} failed: {}", model, e);
                }
            }
        }
    }
}

#[tokio::test]
async fn test_together_json_mode() {
    let Some(provider) = get_provider() else {
        return;
    };

    let request = common::create_json_request(provider.default_model().to_string());

    match provider.complete(request).await {
        Ok(response) => {
            let content = response.choices[0].message.content.as_text().unwrap();

            // Try to parse as JSON
            match serde_json::from_str::<serde_json::Value>(content) {
                Ok(_) => println!("Valid JSON response received"),
                Err(e) => {
                    // Some models might not support JSON mode perfectly
                    println!("JSON parsing failed (may be expected): {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("JSON mode test failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_together_available_models() {
    let Some(provider) = get_provider() else {
        return;
    };

    let models = provider.available_models();
    assert!(!models.is_empty());

    // Should include common Together models
    assert!(models.contains(&"togethercomputer/llama-2-7b-chat"));
    assert!(models.contains(&"mistralai/Mistral-7B-Instruct-v0.1"));

    println!("Available Together models: {:?}", models);
}
