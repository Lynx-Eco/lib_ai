use lib_ai::{providers::CohereProvider, CompletionProvider, AiError};
use std::env;
use dotenvy::dotenv;

mod common;

fn get_provider() -> Option<CohereProvider> {
    dotenv().ok();
    
    if let Ok(api_key) = env::var("COHERE_API_KEY") {
        match CohereProvider::new(api_key) {
            Ok(provider) => Some(provider),
            Err(e) => {
                eprintln!("Failed to create Cohere provider: {}", e);
                None
            }
        }
    } else {
        eprintln!("Skipping Cohere tests: COHERE_API_KEY not set");
        None
    }
}

#[tokio::test]
async fn test_cohere_simple_completion() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_simple_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert!(!response.id.is_empty());
    assert_eq!(response.model, provider.default_model());
    assert!(!response.choices.is_empty());
    assert!(response.choices[0].message.content.as_text().is_some());
    
    // Check usage
    assert!(response.usage.is_some());
    let usage = response.usage.unwrap();
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
}

#[tokio::test]
async fn test_cohere_streaming() {
    let Some(provider) = get_provider() else { return };
    
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
async fn test_cohere_temperature() {
    let Some(provider) = get_provider() else { return };
    
    // Test with temperature 0 for deterministic output
    let mut request = common::create_simple_request(provider.default_model().to_string());
    request.messages[0].content = lib_ai::MessageContent::text("Complete this: 2 + 2 equals");
    request.temperature = Some(0.0);
    request.max_tokens = Some(10);
    
    let response1 = provider.complete(request.clone()).await.unwrap();
    let text1 = response1.choices[0].message.content.as_text().unwrap();
    
    // Should be deterministic
    let response2 = provider.complete(request).await.unwrap();
    let text2 = response2.choices[0].message.content.as_text().unwrap();
    
    // With temperature 0, responses should be very similar or identical
    println!("Response 1: {}", text1);
    println!("Response 2: {}", text2);
}

#[tokio::test]
async fn test_cohere_conversation() {
    let Some(provider) = get_provider() else { return };
    
    let request = common::create_conversation_request(provider.default_model().to_string());
    let response = provider.complete(request).await.unwrap();
    
    assert!(!response.choices.is_empty());
    let content = response.choices[0].message.content.as_text().unwrap();
    
    // Should remember the name from the conversation
    assert!(content.to_lowercase().contains("alice") || 
           content.contains("earlier") || 
           content.contains("mentioned"));
}

#[tokio::test]
async fn test_cohere_error_handling() {
    let provider = match CohereProvider::new("invalid-key".to_string()) {
        Ok(provider) => provider,
        Err(_) => return, // Skip if provider creation fails
    };
    
    let request = common::create_simple_request(provider.default_model().to_string());
    let result = provider.complete(request).await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        AiError::InvalidApiKey { provider } => {
            assert_eq!(provider, "cohere");
        }
        AiError::AuthenticationFailed { .. } => {
            // Also acceptable
        }
        AiError::ProviderError { provider, .. } => {
            assert_eq!(provider, "cohere");
        }
        e => panic!("Unexpected error type: {:?}", e),
    }
}

#[tokio::test]
async fn test_cohere_model_switching() {
    let Some(provider) = get_provider() else { return };
    
    let models = provider.available_models();
    assert!(!models.is_empty());
    
    // Test with different model if available
    if models.len() > 1 {
        let alt_model = models[1];
        let mut request = common::create_simple_request(alt_model.to_string());
        request.max_tokens = Some(50);
        
        match provider.complete(request).await {
            Ok(response) => {
                assert_eq!(response.model, alt_model);
            }
            Err(AiError::UnsupportedModel { .. }) => {
                // Model might not be available on the account
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}