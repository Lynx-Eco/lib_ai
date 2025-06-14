use lib_ai::{
    providers::OllamaProvider, CompletionProvider, CompletionRequest, Message, MessageContent, Role,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Create Ollama provider (connects to local instance)
    let provider = OllamaProvider::new(
        None,                       // Use default URL (http://localhost:11434)
        Some("llama2".to_string()), // Specify model
    );

    println!("🦙 Ollama Local LLM Example");
    println!("==========================");
    println!("Make sure Ollama is running locally with: ollama serve");
    println!("And you have pulled a model with: ollama pull llama2");

    // Basic completion
    let request = CompletionRequest {
        model: "llama2".to_string(), // You can use any model you have locally
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful AI assistant running locally."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text(
                    "What are the benefits of running AI models locally?",
                ),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(500),
        stream: Some(false),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        json_schema: None,
    };

    println!("\n📝 Sending request to local Ollama...");

    match provider.complete(request.clone()).await {
        Ok(response) => {
            if let Some(text) = response.choices[0].message.content.as_text() {
                println!("\n📖 Response:");
                println!("{}", text);
            }

            // Token usage
            if let Some(usage) = response.usage {
                println!("\n📊 Token Usage:");
                println!("  Prompt tokens: {}", usage.prompt_tokens);
                println!("  Completion tokens: {}", usage.completion_tokens);
                println!(
                    "  Total tokens: {}",
                    usage.total_tokens
                );
            }
        }
        Err(e) => {
            eprintln!("\n❌ Error: {}", e);
            eprintln!("\nMake sure:");
            eprintln!("1. Ollama is installed: https://ollama.ai");
            eprintln!("2. Ollama is running: ollama serve");
            eprintln!("3. You have a model: ollama pull llama2");
            return Err(e.into());
        }
    }

    // Streaming example
    println!("\n🌊 Streaming Example:");
    println!("Explain quantum computing in simple terms:");

    let mut stream_request = request;
    stream_request.stream = Some(true);
    stream_request.messages[1].content =
        MessageContent::text("Explain quantum computing in simple terms");

    use futures::StreamExt;
    match provider.complete_stream(stream_request).await {
        Ok(mut stream) => {
            print!("\n");
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => {
                        for choice in chunk.choices {
                            if let Some(content) = &choice.delta.content {
                                print!("{}", content);
                                use std::io::{self, Write};
                                io::stdout().flush()?;
                            }
                        }
                    }
                    Err(e) => eprintln!("\nError in stream: {}", e),
                }
            }
        }
        Err(e) => {
            eprintln!("\n❌ Streaming error: {}", e);
        }
    }

    // List available models
    println!("\n\n📋 Available models:");
    for model in provider.available_models() {
        println!("  - {}", model);
    }

    println!("\n✅ Example completed!");

    Ok(())
}
