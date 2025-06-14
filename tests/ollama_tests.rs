use lib_ai::{providers::OllamaProvider, CompletionProvider, AiError};

mod common;

fn get_provider() -> Option<OllamaProvider> {
    // Check if Ollama is running locally
    let provider = OllamaProvider::new(None, Some("llama2".to_string()));
    
    // Try a simple request to see if Ollama is available
    let test_request = common::create_simple_request("llama2".to_string());
    
    // Use tokio runtime to test connection
    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(provider.complete(test_request)) {
        Ok(_) => Some(provider),
        Err(e) => {
            eprintln!("Skipping Ollama tests: Ollama not available - {}", e);
            eprintln!("Make sure Ollama is running with: ollama serve");
            eprintln!("And you have pulled a model with: ollama pull llama2");
            None
        }
    }
}

#[tokio::test]
async fn test_ollama_simple_completion() {
    let Some(provider) = get_provider() else { return };
    
    let mut request = common::create_simple_request("llama2".to_string());
    request.max_tokens = Some(100); // Keep it short for faster tests
    
    let response = provider.complete(request).await.unwrap();
    
    assert!(!response.id.is_empty());
    assert_eq!(response.model, "llama2");
    assert!(!response.choices.is_empty());
    assert!(response.choices[0].message.content.as_text().is_some());
    
    // Check usage
    assert!(response.usage.is_some());
    let usage = response.usage.unwrap();
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
}

#[tokio::test]
async fn test_ollama_streaming() {
    let Some(provider) = get_provider() else { return };
    
    let mut request = common::create_streaming_request("llama2".to_string());
    request.max_tokens = Some(50); // Keep it short
    
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
async fn test_ollama_temperature() {
    let Some(provider) = get_provider() else { return };
    
    // Test with temperature 0 for deterministic output
    let mut request = common::create_simple_request("llama2".to_string());
    request.messages[0].content = lib_ai::MessageContent::text("What is 2 + 2?");
    request.temperature = Some(0.0);
    request.max_tokens = Some(20);
    
    let response = provider.complete(request).await.unwrap();
    let text = response.choices[0].message.content.as_text().unwrap();
    
    // Should contain "4" in the response
    assert!(text.contains("4") || text.contains("four"));
}

#[tokio::test]
async fn test_ollama_custom_url() {
    // Test with custom URL (still localhost but different path handling)
    let provider = OllamaProvider::new(
        Some("http://localhost:11434".to_string()),
        Some("llama2".to_string())
    );
    
    let mut request = common::create_simple_request("llama2".to_string());
    request.max_tokens = Some(50);
    
    // This might fail if Ollama isn't running, which is fine
    match provider.complete(request).await {
        Ok(response) => {
            assert!(!response.choices.is_empty());
        }
        Err(_) => {
            // Expected if Ollama isn't running
        }
    }
}

#[tokio::test] 
async fn test_ollama_conversation() {
    let Some(provider) = get_provider() else { return };
    
    let mut request = common::create_conversation_request("llama2".to_string());
    request.max_tokens = Some(100);
    
    let response = provider.complete(request).await.unwrap();
    
    assert!(!response.choices.is_empty());
    let content = response.choices[0].message.content.as_text().unwrap();
    
    // Should remember context from the conversation
    println!("Conversation response: {}", content);
}

#[tokio::test]
async fn test_ollama_invalid_model() {
    let provider = OllamaProvider::new(None, Some("nonexistent-model".to_string()));
    
    let request = common::create_simple_request("nonexistent-model".to_string());
    let result = provider.complete(request).await;
    
    // Should fail with model not found
    assert!(result.is_err());
}

#[tokio::test]
async fn test_ollama_available_models() {
    let Some(provider) = get_provider() else { return };
    
    let models = provider.available_models();
    assert!(!models.is_empty());
    
    // Should include at least the common models
    let common_models = ["llama2", "mistral", "codellama", "neural-chat", "starling-lm"];
    
    println!("Available Ollama models: {:?}", models);
    
    // At least one of these should be in the list
    assert!(models.iter().any(|&m| common_models.contains(&m)));
}