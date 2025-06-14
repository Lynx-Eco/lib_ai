use futures::StreamExt;
use lib_ai::{agent::AgentBuilder, providers::OpenAIProvider};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create a provider
    let provider = OpenAIProvider::new(api_key);

    // Build a streaming agent
    let agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are a creative storyteller. Tell engaging stories.")
        .model("gpt-4o-mini")
        .temperature(0.9)
        .max_tokens(500)
        .stream(true) // Enable streaming
        .build()?;

    let mut agent = agent;

    // Request a story with streaming response
    println!("User: Tell me a short story about a robot learning to paint");
    println!("\nAssistant: ");

    let mut stream = agent
        .execute_stream("Tell me a short story about a robot learning to paint")
        .await?;

    // Print the response as it streams
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => print!("{}", text),
            Err(e) => eprintln!("\nError: {}", e),
        }
    }
    println!("\n");

    Ok(())
}
