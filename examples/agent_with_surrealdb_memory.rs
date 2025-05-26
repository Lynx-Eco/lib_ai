use lib_ai::agent::{AgentBuilder};
use lib_ai::agent::memory::{SurrealMemoryStore, SurrealMemoryConfig};
use lib_ai::embeddings::OpenAIEmbeddingProvider;
use lib_ai::providers::OpenAIProvider;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
    
    // Create OpenAI provider
    let provider = OpenAIProvider::new(api_key.clone());
    
    // Create OpenAI embedding provider
    let embedding_provider = Box::new(OpenAIEmbeddingProvider::new(api_key));
    
    // Note: SurrealDB connection will be handled internally by SurrealMemoryStore
    
    // Configure SurrealDB memory
    let memory_config = SurrealMemoryConfig {
        url: "ws://localhost:8000".to_string(),
        namespace: "agent_demo".to_string(),
        database: "memory".to_string(),
        table: "conversations".to_string(),
        username: None,
        password: None,
    };
    
    // Create SurrealDB memory store
    let memory_store = SurrealMemoryStore::new(memory_config, embedding_provider)
        .await
        .expect("Failed to create SurrealDB memory store. Make sure SurrealDB is running on localhost:8000");
    
    // Build agent with SurrealDB memory
    let mut agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are a helpful assistant with the ability to remember past conversations. \
                 Use your memory to provide context-aware responses.")
        .memory(memory_store)
        .build()
        .expect("Failed to build agent");
    
    println!("Agent with SurrealDB Memory Example");
    println!("=====================================");
    println!("This agent remembers conversations and can search through past interactions.");
    println!("Try asking questions, then referring back to previous topics!");
    println!();
    
    // Example conversation loop
    let conversations = vec![
        "Hi! I'm interested in learning about machine learning.",
        "What are the main types of machine learning?",
        "Can you explain supervised learning in more detail?",
        "What did I first ask you about?", // This should trigger memory recall
        "Search your memory for information about machine learning", // This should trigger semantic search
    ];
    
    for user_input in conversations {
        println!("User: {}", user_input);
        
        // Get response from agent
        let response = agent.chat(user_input).await?;
        println!("Agent: {}", response);
        println!();
        
        // Small delay for readability
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
    
    println!("\nConversation complete!");
    println!("The agent's memory is persisted in SurrealDB and will be available in future sessions.");
    
    Ok(())
}