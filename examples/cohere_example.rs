use dotenv::dotenv;
use lib_ai::{
    providers::CohereProvider, CompletionProvider, CompletionRequest, Message, MessageContent, Role,
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Get API key from environment
    let api_key =
        env::var("COHERE_API_KEY").expect("Please set COHERE_API_KEY environment variable");

    // Create Cohere provider
    let provider = CohereProvider::new(Some(api_key))?;

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
