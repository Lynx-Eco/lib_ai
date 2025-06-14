use lib_ai::agent::memory::{Memory, SurrealMemoryConfig, SurrealMemoryStore};
use lib_ai::embeddings::MockEmbeddingProvider;

#[tokio::test]
#[ignore] // Requires SurrealDB to be running
async fn test_surrealdb_memory_initialization() {
    // Create mock embedding provider
    let embedding_provider = Box::new(MockEmbeddingProvider::new(384));

    // Create config
    let config = SurrealMemoryConfig {
        url: "ws://localhost:8000".to_string(),
        namespace: "test".to_string(),
        database: "memory".to_string(),
        table: "conversations".to_string(),
        username: None,
        password: None,
    };

    // Create memory store
    let memory = SurrealMemoryStore::new(config, embedding_provider)
        .await
        .unwrap();

    // Test stats
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 0);
}

#[tokio::test]
#[ignore] // Requires SurrealDB to be running
async fn test_surrealdb_memory_store_and_retrieve() {
    let embedding_provider = Box::new(MockEmbeddingProvider::new(384));
    let config = SurrealMemoryConfig::default();

    let mut memory = SurrealMemoryStore::new(config, embedding_provider)
        .await
        .unwrap();

    // Store a conversation
    let input = "What is the capital of France?";
    let output = "The capital of France is Paris.";

    memory.store(input, output).await.unwrap();

    // Test retrieve
    let retrieved = memory.retrieve("capital of France", 1).await.unwrap();
    assert!(!retrieved.is_empty());
    assert!(retrieved[0].contains("France") || retrieved[0].contains("Paris"));

    // Test stats
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 1);
}

#[tokio::test]
#[ignore] // Requires SurrealDB to be running
async fn test_surrealdb_memory_vector_search() {
    let embedding_provider = Box::new(MockEmbeddingProvider::with_similarity());
    let config = SurrealMemoryConfig::default();

    let mut memory = SurrealMemoryStore::new(config, embedding_provider)
        .await
        .unwrap();

    // Store multiple conversations
    let conversations = vec![
        (
            "Tell me about machine learning",
            "Machine learning is a subset of artificial intelligence",
        ),
        (
            "What's the weather like today?",
            "I don't have access to current weather data",
        ),
        (
            "Explain neural networks",
            "Neural networks are computing systems inspired by biological neural networks",
        ),
    ];

    for (input, output) in conversations {
        memory.store(input, output).await.unwrap();
    }

    // Search for ML-related content
    let results = memory.retrieve("artificial intelligence", 2).await.unwrap();
    assert!(!results.is_empty());
    assert!(
        results[0].contains("machine learning")
            || results[0].contains("artificial intelligence")
            || results[0].contains("neural")
    );
}

#[tokio::test]
#[ignore] // Requires SurrealDB to be running
async fn test_surrealdb_memory_clear() {
    let embedding_provider = Box::new(MockEmbeddingProvider::new(384));
    let config = SurrealMemoryConfig::default();

    let mut memory = SurrealMemoryStore::new(config, embedding_provider)
        .await
        .unwrap();

    // Store some entries
    for i in 0..5 {
        memory
            .store(&format!("Test message {}", i), &format!("Response {}", i))
            .await
            .unwrap();
    }

    // Verify entries exist
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 5);

    // Clear memory
    memory.clear().await.unwrap();

    // Verify cleared
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.total_entries, 0);
}

#[tokio::test]
#[ignore] // Requires SurrealDB to be running
async fn test_surrealdb_memory_persistence() {
    let embedding_provider = Box::new(MockEmbeddingProvider::new(384));
    let config = SurrealMemoryConfig {
        url: "ws://localhost:8000".to_string(),
        namespace: "test_persistence".to_string(),
        database: "memory".to_string(),
        table: "conversations_persist".to_string(),
        username: None,
        password: None,
    };

    // Create first instance and store data
    {
        let mut memory =
            SurrealMemoryStore::new(config.clone(), Box::new(MockEmbeddingProvider::new(384)))
                .await
                .unwrap();
        memory
            .store("What is Rust?", "Rust is a systems programming language")
            .await
            .unwrap();
    }

    // Create second instance and verify data persists
    {
        let memory = SurrealMemoryStore::new(config, embedding_provider)
            .await
            .unwrap();
        let results = memory.retrieve("Rust programming", 1).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].contains("Rust") || results[0].contains("systems"));
    }
}
