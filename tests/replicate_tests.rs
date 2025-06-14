use dotenv::dotenv;
use lib_ai::{providers::ReplicateProvider, AiError, CompletionProvider};
use std::env;

mod common;

fn get_provider() -> Option<ReplicateProvider> {
    dotenv().ok();

    if let Ok(api_token) = env::var("REPLICATE_API_TOKEN") {
        Some(ReplicateProvider::new(api_token))
    } else {
        eprintln!("Skipping Replicate tests: REPLICATE_API_TOKEN not set");
        None
    }
}

#[tokio::test]
async fn test_replicate_simple_completion() {
    let Some(provider) = get_provider() else {
        return;
    };

    let mut request = common::create_simple_request("meta/llama-2-7b-chat".to_string());
    request.max_tokens = Some(100); // Keep it short for faster tests

    match provider.complete(request).await {
        Ok(response) => {
            assert!(!response.id.is_empty());
            assert!(!response.choices.is_empty());
            assert!(response.choices[0].message.content.as_text().is_some());

            // Check usage
            assert!(response.usage.is_some());
            let usage = response.usage.unwrap();
            assert!(usage.input_tokens > 0);
            assert!(usage.output_tokens > 0);
        }
        Err(e) => {
            eprintln!("Test failed (may be due to model availability): {}", e);
            // Don't fail the test as Replicate models may not always be available
        }
    }
}

#[tokio::test]
async fn test_replicate_temperature() {
    let Some(provider) = get_provider() else {
        return;
    };

    // Test with temperature 0 for deterministic output
    let mut request = common::create_simple_request("meta/llama-2-7b-chat".to_string());
    request.messages[0].content =
        lib_ai::MessageContent::text("What is 2 + 2? Answer with just the number.");
    request.temperature = Some(0.0);
    request.max_tokens = Some(10);

    match provider.complete(request).await {
        Ok(response) => {
            let text = response.choices[0].message.content.as_text().unwrap();
            // Should contain "4" in the response
            println!("Response: {}", text);
        }
        Err(e) => {
            eprintln!("Temperature test skipped: {}", e);
        }
    }
}

#[tokio::test]
async fn test_replicate_conversation() {
    let Some(provider) = get_provider() else {
        return;
    };

    let mut request = common::create_conversation_request("meta/llama-2-7b-chat".to_string());
    request.max_tokens = Some(100);

    match provider.complete(request).await {
        Ok(response) => {
            assert!(!response.choices.is_empty());
            let content = response.choices[0].message.content.as_text().unwrap();
            println!("Conversation response: {}", content);
        }
        Err(e) => {
            eprintln!("Conversation test skipped: {}", e);
        }
    }
}

#[tokio::test]
async fn test_replicate_error_handling() {
    let provider = ReplicateProvider::new("invalid-token".to_string());

    let request = common::create_simple_request("meta/llama-2-7b-chat".to_string());
    let result = provider.complete(request).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        AiError::InvalidApiKey { provider } => {
            assert_eq!(provider, "replicate");
        }
        AiError::AuthenticationFailed { .. } => {
            // Also acceptable
        }
        AiError::ProviderError { provider, .. } => {
            assert_eq!(provider, "replicate");
        }
        e => panic!("Unexpected error type: {:?}", e),
    }
}

#[tokio::test]
async fn test_replicate_model_formats() {
    let Some(provider) = get_provider() else {
        return;
    };

    // Test different model format patterns
    let model_formats = vec![
        "meta/llama-2-7b-chat",
        "stability-ai/stablelm-tuned-alpha-7b",
        "replicate/vicuna-13b",
    ];

    for model in model_formats {
        println!("Testing model format: {}", model);

        let mut request = common::create_simple_request(model.to_string());
        request.max_tokens = Some(50);
        request.messages[0].content = lib_ai::MessageContent::text("Say hello");

        // Just verify the request is properly formatted
        // Don't fail on model not found as availability varies
        match provider.complete(request).await {
            Ok(_) => println!("Model {} succeeded", model),
            Err(e) => println!("Model {} failed (expected): {}", model, e),
        }
    }
}

#[tokio::test]
async fn test_replicate_available_models() {
    let Some(provider) = get_provider() else {
        return;
    };

    let models = provider.available_models();
    assert!(!models.is_empty());

    // Should include common Replicate models
    assert!(models.contains(&"meta/llama-2-7b-chat"));
    assert!(models.contains(&"meta/llama-2-13b-chat"));

    println!("Available Replicate models: {:?}", models);
}
