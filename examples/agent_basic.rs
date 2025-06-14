use lib_ai::{
    agent::{tools::CalculatorTool, AgentBuilder, InMemoryStore},
    providers::OpenAIProvider,
};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create a provider
    let provider = OpenAIProvider::new(api_key);

    // Build an agent with a calculator tool
    let agent = AgentBuilder::new()
        .provider(provider)
        .prompt(
            "You are a helpful assistant with access to a calculator. Use it when needed for math.",
        )
        .model("gpt-4o-mini")
        .temperature(0.7)
        .tool("calculator", CalculatorTool)
        .memory(InMemoryStore::new(100))
        .build()?;

    // Use the agent
    let mut agent = agent;

    // First interaction
    println!("User: What is 42 * 17?");
    let response = agent.execute("What is 42 * 17?").await?;
    println!("Assistant: {}\n", response);

    // Second interaction (will use memory)
    println!("User: What was the result of the previous calculation?");
    let response = agent
        .execute("What was the result of the previous calculation?")
        .await?;
    println!("Assistant: {}\n", response);

    // Third interaction with multiple calculations
    println!("User: Now divide that by 2 and add 100");
    let response = agent.execute("Now divide that by 2 and add 100").await?;
    println!("Assistant: {}", response);

    Ok(())
}
