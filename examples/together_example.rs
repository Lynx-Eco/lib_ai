use dotenv::dotenv;
use lib_ai::{
    providers::TogetherProvider, CompletionProvider, CompletionRequest, Message, MessageContent,
    Role,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Try to create Together provider
    let provider = match TogetherProvider::new(None) {
        Ok(p) => p,
        Err(_) => {
            eprintln!("❌ Error: TOGETHER_API_KEY environment variable not set.");
            eprintln!("   Please set it with: export TOGETHER_API_KEY=your-api-key");
            eprintln!("   Or create a .env file with TOGETHER_API_KEY=your-api-key");
            return Ok(());
        }
    };

    println!("🤝 Together AI Provider Example");
    println!("==============================");
    println!("Using open source models via Together AI\n");

    // List available models
    println!("📋 Available models:");
    for model in provider.available_models() {
        println!("  - {}", model);
    }

    // Basic completion with Llama 2
    let request = CompletionRequest {
        model: "togethercomputer/llama-2-7b-chat".to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful AI assistant."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("What are the advantages of open source AI models?"),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(300),
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

    println!("\n📝 Sending request to Together AI (Llama 2)...");
    let response = provider.complete(request.clone()).await?;

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

    // Try another model - Code Llama
    println!("\n🔧 Trying Code Llama for code generation:");
    let mut code_request = request;
    code_request.model = "togethercomputer/CodeLlama-7b-Instruct".to_string();
    code_request.messages[1].content = MessageContent::text(
        "Write a Python function to calculate the Fibonacci sequence recursively with memoization",
    );

    let code_response = provider.complete(code_request).await?;

    if let Some(text) = code_response.choices[0].message.content.as_text() {
        println!("\n📖 Code Response:");
        println!("{}", text);
    }

    // Streaming example
    println!("\n🌊 Streaming Example with Mistral:");
    let stream_request = CompletionRequest {
        model: "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::text(
                "Explain the concept of machine learning in simple terms",
            ),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: Some(0.7),
        max_tokens: Some(200),
        stream: Some(true),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        json_schema: None,
    };

    use futures::StreamExt;
    let mut stream = provider.complete_stream(stream_request).await?;

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

    println!("\n\n✅ Together AI example completed!");

    Ok(())
}
