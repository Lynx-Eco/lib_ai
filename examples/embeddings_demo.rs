use lib_ai::agent::memory::{Memory, SemanticMemoryBuilder};
use lib_ai::embeddings::{
    EmbeddingProvider, EmbeddingRequest, MockEmbeddingProvider, OpenAIEmbeddingProvider,
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Embeddings and Memory Demo");
    println!("==========================\n");
    dotenv::dotenv().ok();
    // You can switch between providers
    let use_openai = env::var("OPENAI_API_KEY").is_ok();

    let embedding_provider: Box<dyn EmbeddingProvider> = if use_openai {
        println!("Using OpenAI embeddings (text-embedding-3-small)");
        let api_key = env::var("OPENAI_API_KEY").unwrap();
        Box::new(OpenAIEmbeddingProvider::new(api_key))
    } else {
        println!("Using mock embeddings (set OPENAI_API_KEY to use real embeddings)");
        Box::new(MockEmbeddingProvider::new(384))
    };

    // Demo 1: Direct embedding generation
    println!("\n1. Direct Embedding Generation:");
    println!("-------------------------------");

    let texts = vec![
        "Machine learning is amazing",
        "Deep learning uses neural networks",
        "The weather is sunny today",
    ];

    let request = EmbeddingRequest {
        input: texts.iter().map(|s| s.to_string()).collect(),
        model: embedding_provider.default_model().to_string(),
    };

    let response = embedding_provider.embed(request).await?;

    for (text, embedding) in texts.iter().zip(response.embeddings.iter()) {
        println!("Text: \"{}\"", text);
        println!("Embedding dimension: {}", embedding.vector.len());
        println!(
            "First 5 values: {:?}\n",
            &embedding.vector[..(5).min(embedding.vector.len())]
        );
    }

    // Demo 2: Cosine similarity
    println!("\n2. Cosine Similarity Demo:");
    println!("--------------------------");

    if response.embeddings.len() >= 3 {
        let ml_embedding = &response.embeddings[0];
        let dl_embedding = &response.embeddings[1];
        let weather_embedding = &response.embeddings[2];

        let ml_dl_similarity = ml_embedding.cosine_similarity(dl_embedding);
        let ml_weather_similarity = ml_embedding.cosine_similarity(weather_embedding);

        println!(
            "Similarity between \"{}\" and \"{}\"): {:.3}",
            texts[0], texts[1], ml_dl_similarity
        );
        println!(
            "Similarity between \"{}\" and \"{}\"): {:.3}",
            texts[0], texts[2], ml_weather_similarity
        );
        println!("\nAs expected, ML and DL are more similar than ML and weather!");
    }

    // Demo 3: Semantic memory with embeddings
    println!("\n\n3. Semantic Memory Demo:");
    println!("------------------------");

    let mut memory = SemanticMemoryBuilder::new()
        .embedding_provider_arc(std::sync::Arc::from(embedding_provider))
        .max_entries(100)
        .similarity_threshold(0.5)
        .build()?;

    // Store some knowledge
    let knowledge_base = vec![
        (
            "What is Rust?",
            "Rust is a systems programming language focused on safety and performance",
        ),
        (
            "Tell me about Python",
            "Python is a high-level language popular for data science and web development",
        ),
        (
            "Explain JavaScript",
            "JavaScript is a dynamic language that runs in browsers and Node.js",
        ),
        (
            "What is machine learning?",
            "Machine learning is a subset of AI that enables computers to learn from data",
        ),
        (
            "What's the weather like?",
            "I don't have access to current weather data",
        ),
    ];

    println!("Storing knowledge base...");
    for (question, answer) in &knowledge_base {
        memory.store(question, answer).await?;
    }

    // Query the memory
    let queries = vec![
        "programming languages",
        "AI and ML",
        "web development",
        "weather forecast",
    ];

    println!("\nQuerying semantic memory:");
    for query in queries {
        println!("\nQuery: \"{}\"", query);
        let results = memory.retrieve(query, 2).await?;

        if results.is_empty() {
            println!("  No relevant results found");
        } else {
            for (i, result) in results.iter().enumerate() {
                println!("  Result {}:", i + 1);
                for line in result.lines() {
                    println!("    {}", line);
                }
            }
        }
    }

    // Show statistics
    let stats = memory.stats().await?;
    println!("\n\nMemory Statistics:");
    println!("------------------");
    println!("Total entries: {}", stats.total_entries);
    println!("Total size: {} bytes", stats.total_size_bytes);

    Ok(())
}
