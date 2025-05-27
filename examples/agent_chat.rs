use lib_ai::{
    agent::{AgentBuilder, InMemoryStore},
    providers::AnthropicProvider,
};
use std::io::{self, Write};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    // Get API key from environment
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("Please set ANTHROPIC_API_KEY environment variable");
    
    // Create a provider
    let provider = AnthropicProvider::new(api_key);
    
    // Build a conversational agent
    let agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are Claude, a helpful AI assistant. Be concise but friendly.")
        .model("claude-3-5-haiku-20241022")
        .temperature(0.7)
        .memory(InMemoryStore::new(50))
        .max_iterations(5)  // Allow up to 5 tool uses per turn
        .build()?;
    
    let mut agent = agent;
    
    println!("Chat with Claude (type 'quit' to exit, 'clear' to reset context)");
    println!("{}", "=".repeat(60));
    
    loop {
        // Get user input
        print!("\nYou: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        // Check for commands
        match input.to_lowercase().as_str() {
            "quit" | "exit" => {
                println!("Goodbye!");
                break;
            }
            "clear" => {
                agent.clear_context();
                println!("[Context cleared]");
                continue;
            }
            _ => {}
        }
        
        // Get response from agent
        print!("\nClaude: ");
        match agent.chat(input).await {
            Ok(response) => println!("{}", response),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    Ok(())
}