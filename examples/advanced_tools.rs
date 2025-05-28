use lib_ai::{
    agent::{AgentBuilder, FileSystemTool, HttpTool, CodeExecutorTool},
    providers::OpenAIProvider,
};
use std::error::Error;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();
    
    // Create provider
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set");
    let provider = OpenAIProvider::new(api_key);
    
    // Create a temporary directory for file operations
    let temp_dir = TempDir::new()?;
    println!("Working directory: {:?}", temp_dir.path());
    
    // Create tools
    let fs_tool = FileSystemTool::new(temp_dir.path())
        .with_write_access()
        .with_max_file_size(1024 * 1024); // 1MB
    
    let http_tool = HttpTool::new()
        .with_max_response_size(1024 * 1024) // 1MB
        .add_allowed_domain("api.github.com")
        .add_allowed_domain("jsonplaceholder.typicode.com");
    
    let code_tool = CodeExecutorTool::new()
        .with_timeout(10)
        .add_allowed_language("python")
        .add_allowed_language("javascript");
    
    // Create agent with tools
    let mut agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are a helpful assistant with access to file system, HTTP, and code execution tools.")
        .tool("filesystem", fs_tool)
        .tool("http", http_tool)
        .tool("code_executor", code_tool)
        .build()?;
    
    // Example 1: File system operations
    println!("\n=== File System Operations ===");
    let response = agent.execute("Create a file called 'test.txt' with the content 'Hello from AI Agent!'").await?;
    println!("Agent: {}", response);
    
    let response = agent.execute("Now read the contents of test.txt").await?;
    println!("Agent: {}", response);
    
    // Example 2: HTTP requests
    println!("\n=== HTTP Requests ===");
    let response = agent.execute("Fetch information about the Rust programming language repository from GitHub API (rust-lang/rust)").await?;
    println!("Agent: {}", response);
    
    // Example 3: Code execution
    println!("\n=== Code Execution ===");
    let response = agent.execute("Write and execute a Python script that calculates the first 10 Fibonacci numbers").await?;
    println!("Agent: {}", response);
    
    let response = agent.execute("Now do the same in JavaScript").await?;
    println!("Agent: {}", response);
    
    // Example 4: Combined operations
    println!("\n=== Combined Operations ===");
    let response = agent.execute("
        1. Fetch the current Bitcoin price from a public API
        2. Save the price to a file called 'bitcoin_price.json'
        3. Write a Python script that reads this file and displays the price
        4. Execute the script
    ").await?;
    println!("Agent: {}", response);
    
    Ok(())
}