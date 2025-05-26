use lib_ai::agent::{AgentBuilder};
use lib_ai::agent::memory::{SemanticMemoryBuilder};
use lib_ai::embeddings::MockEmbeddingProvider;
use lib_ai::providers::OpenAIProvider;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
    
    // Create OpenAI provider
    let provider = OpenAIProvider::new(api_key);
    
    // Create semantic memory with mock embeddings (for demonstration)
    // In production, you'd use a real embedding provider like OpenAIEmbeddingProvider
    let memory = SemanticMemoryBuilder::new()
        .embedding_provider(MockEmbeddingProvider::with_similarity())
        .max_entries(1000)
        .similarity_threshold(0.7)
        .build()?;
    
    // Build agent with semantic memory
    let mut agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are a helpful assistant with semantic memory. \
                 You can remember past conversations and find relevant information \
                 based on meaning, not just keywords.")
        .memory(memory)
        .build()
        .expect("Failed to build agent");
    
    println!("Agent with Semantic Memory Example");
    println!("===================================");
    println!("This agent uses vector embeddings to find semantically similar memories.");
    println!();
    
    // Example conversation demonstrating semantic search
    let conversations = vec![
        "I love programming in Rust because of its memory safety.",
        "Python is great for data science and machine learning.",
        "JavaScript is essential for web development.",
        "What programming languages have I mentioned?", // Should recall all three
        "Which language did I mention is good for safety?", // Should find Rust via semantic similarity
        "Tell me what I said about web technologies.", // Should find JavaScript
    ];
    
    for user_input in conversations {
        println!("User: {}", user_input);
        
        // Get response from agent
        let response = agent.chat(user_input).await?;
        println!("Agent: {}", response);
        println!();
        
        // Small delay for readability
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
    
    println!("\nConversation complete!");
    println!("Note: The agent's memory system stores conversations for context-aware responses.");
    
    Ok(())
}