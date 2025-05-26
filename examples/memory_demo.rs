use lib_ai::agent::memory::{Memory, InMemoryStore, SemanticMemoryBuilder};
use lib_ai::embeddings::MockEmbeddingProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory System Demo");
    println!("==================");
    println!();

    // Demo 1: Simple In-Memory Store
    println!("1. Simple In-Memory Store:");
    println!("--------------------------");
    
    let mut simple_memory = InMemoryStore::new(10);
    
    // Store some conversations
    simple_memory.store("What is Rust?", "Rust is a systems programming language").await?;
    simple_memory.store("Tell me about Python", "Python is great for data science").await?;
    simple_memory.store("Explain JavaScript", "JavaScript runs in browsers").await?;
    
    // Retrieve based on keywords
    println!("\nStored 3 conversations. Now searching:");
    
    let queries = vec!["Rust", "Python", "programming", "browsers"];
    for query in queries {
        let results = simple_memory.retrieve(query, 2).await?;
        println!("\nQuery: '{}'", query);
        if results.is_empty() {
            println!("  No results found");
        } else {
            for (i, result) in results.iter().enumerate() {
                println!("  {}. {}", i + 1, result);
            }
        }
    }
    
    // Show stats
    let stats = simple_memory.stats().await?;
    println!("\nMemory Stats:");
    println!("  Total entries: {}", stats.total_entries);
    println!("  Total size: {} bytes", stats.total_size_bytes);
    
    // Demo 2: Semantic Memory
    println!("\n\n2. Semantic Memory with Embeddings:");
    println!("------------------------------------");
    
    let mut semantic_memory = SemanticMemoryBuilder::new()
        .embedding_provider(MockEmbeddingProvider::with_similarity())
        .max_entries(100)
        .similarity_threshold(0.3)
        .build()?;
    
    // Store knowledge base
    let knowledge = vec![
        ("machine learning", "ML is a subset of AI that learns from data"),
        ("deep learning", "DL uses neural networks with multiple layers"),
        ("natural language", "NLP processes and understands human language"),
        ("computer vision", "CV enables computers to understand visual information"),
        ("reinforcement learning", "RL learns through trial and error with rewards"),
    ];
    
    println!("Storing AI/ML knowledge base...");
    for (topic, description) in &knowledge {
        semantic_memory.store(topic, description).await?;
    }
    
    // Semantic search
    println!("\nSemantic search results:");
    let search_queries = vec![
        "neural networks",
        "learning algorithms",
        "visual recognition",
        "language understanding",
    ];
    
    for query in search_queries {
        println!("\nQuery: '{}'", query);
        let results = semantic_memory.retrieve(query, 2).await?;
        if results.is_empty() {
            println!("  No similar entries found");
        } else {
            for (i, result) in results.iter().enumerate() {
                println!("  {}. {}", i + 1, result);
            }
        }
    }
    
    // Demonstrate max entries limit
    println!("\n\n3. Memory Size Management:");
    println!("--------------------------");
    
    let mut limited_memory = SemanticMemoryBuilder::new()
        .embedding_provider(MockEmbeddingProvider::new(384))
        .max_entries(3)
        .build()?;
    
    // Store more than max
    for i in 1..=5 {
        limited_memory.store(
            &format!("Entry {}", i),
            &format!("This is memory entry number {}", i)
        ).await?;
    }
    
    let stats = limited_memory.stats().await?;
    println!("Stored 5 entries with max_entries=3");
    println!("Current entries: {} (oldest entries were removed)", stats.total_entries);
    
    // Verify only recent entries remain
    let all = limited_memory.retrieve("entry", 10).await?;
    println!("\nRemaining entries:");
    for entry in &all {
        println!("  - {}", entry);
    }
    
    Ok(())
}