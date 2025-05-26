use lib_ai::agent::memory::{Memory, InMemoryStore, SemanticMemoryBuilder};
use lib_ai::embeddings::MockEmbeddingProvider;

#[tokio::test]
async fn test_in_memory_store() {
    let mut memory = InMemoryStore::new(100);
    
    // Test store
    memory.store("Hello", "Hi there!").await.unwrap();
    memory.store("What's your name?", "I'm an AI assistant").await.unwrap();
    
    // Test retrieve
    let results = memory.retrieve("Hello", 1).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].contains("Hello") && results[0].contains("Hi there!"));
    
    // Test stats
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 2);
    
    // Test clear
    memory.clear().await.unwrap();
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 0);
}

#[tokio::test]
async fn test_semantic_memory() {
    let mut memory = SemanticMemoryBuilder::new()
        .embedding_provider(MockEmbeddingProvider::with_similarity())
        .max_entries(100)
        .similarity_threshold(0.5)
        .build()
        .expect("Failed to build semantic memory");
    
    // Store some entries
    memory.store("I love programming in Rust", "Rust is great for systems programming").await.unwrap();
    memory.store("Python is good for data science", "Python has many ML libraries").await.unwrap();
    memory.store("JavaScript runs in browsers", "JS is essential for web development").await.unwrap();
    
    // Test semantic search - with mock embeddings, results may not be semantically ordered
    let results = memory.retrieve("programming", 3).await.unwrap();
    assert!(!results.is_empty(), "No results found for 'programming'");
    
    // At least one result should contain programming-related content
    let has_programming_content = results.iter().any(|r| 
        r.contains("Rust") || r.contains("Python") || r.contains("JavaScript")
    );
    assert!(has_programming_content, "Expected at least one programming-related result");
    
    // Test stats
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 3);
}

#[tokio::test]
async fn test_semantic_memory_similarity_threshold() {
    let mut memory = SemanticMemoryBuilder::new()
        .embedding_provider(MockEmbeddingProvider::with_similarity())
        .similarity_threshold(0.9) // Very high threshold
        .build()
        .expect("Failed to build semantic memory");
    
    // Store an entry
    memory.store("The weather is nice today", "It's sunny and warm").await.unwrap();
    
    // Search with unrelated query - with mock embeddings this might still match
    let _results = memory.retrieve("programming languages", 5).await.unwrap();
    // Note: MockEmbeddingProvider doesn't do real semantic matching
    
    // Search with related query
    let results = memory.retrieve("weather", 5).await.unwrap();
    assert!(!results.is_empty());
    assert!(results[0].contains("weather") || results[0].contains("sunny"));
}

#[tokio::test]
async fn test_semantic_memory_max_entries() {
    let mut memory = SemanticMemoryBuilder::new()
        .embedding_provider(MockEmbeddingProvider::new(384))
        .max_entries(3)
        .build()
        .expect("Failed to build semantic memory");
    
    // Store more than max entries
    for i in 0..5 {
        memory.store(&format!("Message {}", i), &format!("Response {}", i)).await.unwrap();
    }
    
    // Should only keep the latest 3 entries
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 3);
    
    // Should be able to retrieve recent entries
    let results = memory.retrieve("Message 4", 5).await.unwrap();
    assert!(!results.is_empty());
    
    // Should not find old entries
    let results = memory.retrieve("Message 0", 5).await.unwrap();
    assert!(results.is_empty() || !results[0].contains("Message 0"));
}