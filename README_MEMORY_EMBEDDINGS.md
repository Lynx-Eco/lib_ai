# Memory and Embeddings System

This document describes the memory and embeddings system implemented in lib_ai.

## Overview

The library now includes a comprehensive memory system with embedding support for semantic search capabilities.

## Components

### 1. Embeddings Module (`src/embeddings/`)

#### Providers:
- **OpenAIEmbeddingProvider**: Supports OpenAI's text-embedding models
  - text-embedding-3-small (1536 dimensions)
  - text-embedding-3-large (3072 dimensions)
  - text-embedding-ada-002 (1536 dimensions)
  
- **MockEmbeddingProvider**: For testing without API calls
  - Configurable dimensions
  - Deterministic embeddings based on text hash

- **LocalEmbeddingProvider**: For self-hosted embedding models
  - Compatible with sentence-transformers servers

#### Core Features:
- Asynchronous embedding generation
- Batch processing support
- Cosine similarity calculations
- Dimension tracking

### 2. Memory System (`src/agent/memory/`)

#### Memory Implementations:

1. **InMemoryStore**: Simple in-memory storage
   - Fast retrieval
   - Keyword-based search
   - Configurable max entries

2. **SemanticMemory**: Vector-based semantic search
   - Uses embeddings for similarity matching
   - Configurable similarity threshold
   - Memory size management

3. **SurrealDBMemoryStore**: Persistent memory with vector search
   - Uses SurrealDB with RecordId
   - Vector similarity search
   - Scalable storage
   - Metadata support

#### Memory Trait:
```rust
#[async_trait]
pub trait Memory: Send + Sync {
    async fn store(&mut self, input: &str, output: &str) -> Result<(), AgentError>;
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>, AgentError>;
    async fn clear(&mut self) -> Result<(), AgentError>;
    async fn stats(&self) -> Result<MemoryStats, AgentError>;
}
```

## Usage Examples

### Basic Embedding Usage:
```rust
use lib_ai::embeddings::{EmbeddingProvider, OpenAIEmbeddingProvider};

let provider = OpenAIEmbeddingProvider::new(api_key);
let embedding = provider.embed_single("Hello world").await?;
```

### Semantic Memory with Agent:
```rust
use lib_ai::agent::{AgentBuilder};
use lib_ai::agent::memory::{SemanticMemoryBuilder};
use lib_ai::embeddings::OpenAIEmbeddingProvider;

let memory = SemanticMemoryBuilder::new()
    .embedding_provider(OpenAIEmbeddingProvider::new(api_key))
    .similarity_threshold(0.7)
    .build()?;

let agent = AgentBuilder::new()
    .provider(Box::new(provider))
    .memory(Box::new(memory))
    .build();
```

### SurrealDB Persistent Memory:
```rust
use lib_ai::agent::memory::{SurrealMemoryStore, SurrealMemoryConfig};

let config = SurrealMemoryConfig {
    url: "ws://localhost:8000".to_string(),
    namespace: "myapp".to_string(),
    database: "memory".to_string(),
    table: "conversations".to_string(),
    username: None,
    password: None,
};

let memory = SurrealMemoryStore::new(config, embedding_provider).await?;
```

## Running Examples

```bash
# Basic embeddings demo
cargo run --example embeddings_demo

# Agent with semantic memory
cargo run --example agent_with_semantic_memory

# Agent with SurrealDB (requires SurrealDB running)
cargo run --example agent_with_surrealdb_memory
```

## Testing

```bash
# Run memory tests
cargo test --test memory_tests

# Run SurrealDB tests (requires SurrealDB)
cargo test --test surrealdb_memory_tests -- --ignored
```

## Future Enhancements

1. Additional embedding providers:
   - Cohere
   - Anthropic (when available)
   - HuggingFace models

2. More vector databases:
   - Qdrant
   - Pinecone
   - Weaviate

3. Advanced features:
   - Embedding caching
   - Incremental indexing
   - Memory compression
   - Hybrid search (keyword + semantic)