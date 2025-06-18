use dotenv::dotenv;
use lib_ai::{
    providers::CohereProvider, CompletionProvider, CompletionRequest, Message, MessageContent, Role,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Try to create Cohere provider
    let provider = match CohereProvider::new(None) {
        Ok(p) => p,
        Err(_) => {
            eprintln!("❌ Error: COHERE_API_KEY environment variable not set.");
            eprintln!("   Please set it with: export COHERE_API_KEY=your-api-key");
            eprintln!("   Or create a .env file with COHERE_API_KEY=your-api-key");
            return Ok(());
        }
    };

    println!("🤖 Cohere Provider Example");
    println!("=======================");

    // Basic completion
    let request = CompletionRequest {
        model: provider.default_model().to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful AI assistant."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("Write a haiku about programming in Rust"),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(150),
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

    println!("\n📝 Sending request to Cohere...");
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

    // Streaming example
    println!("\n🌊 Streaming Example:");
    println!("Write a short story about a robot learning to paint:");

    let mut stream_request = request;
    stream_request.stream = Some(true);
    stream_request.messages[1].content =
        MessageContent::text("Write a short story about a robot learning to paint");

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
    println!("\n\n✅ Example completed!");

    Ok(())
}
