use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use surrealdb::engine::remote::ws::{Client, Ws};
use surrealdb::sql::Datetime;
use surrealdb::RecordId;
use surrealdb::Surreal;

use super::base::{Memory, MemoryStats};
use crate::agent::AgentError;
use crate::embeddings::{Embedding, EmbeddingProvider};

/// A memory entry stored in SurrealDB
#[derive(Debug, Serialize, Deserialize)]
struct MemoryRecord {
    id: Option<RecordId>,
    input: String,
    output: String,
    embedding: Vec<f32>,
    metadata: Option<serde_json::Value>,
    created_at: Datetime,
}

/// Configuration for SurrealDB memory store
#[derive(Clone)]
pub struct SurrealMemoryConfig {
    pub url: String,
    pub namespace: String,
    pub database: String,
    pub table: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl Default for SurrealMemoryConfig {
    fn default() -> Self {
        Self {
            url: "ws://localhost:8000".to_string(),
            namespace: "lib_ai".to_string(),
            database: "memory".to_string(),
            table: "conversations".to_string(),
            username: None,
            password: None,
        }
    }
}

/// SurrealDB-backed memory store with vector search
pub struct SurrealMemoryStore {
    db: Surreal<Client>,
    config: SurrealMemoryConfig,
    embedding_provider: Box<dyn EmbeddingProvider>,
}

impl SurrealMemoryStore {
    /// Create a new SurrealDB memory store
    pub async fn new(
        config: SurrealMemoryConfig,
        embedding_provider: Box<dyn EmbeddingProvider>,
    ) -> Result<Self, AgentError> {
        let db = Surreal::new::<Ws>(&config.url).await.map_err(|e| {
            AgentError::MemoryError(format!("Failed to connect to SurrealDB: {}", e))
        })?;

        // Authenticate if credentials provided
        if let (Some(username), Some(password)) = (&config.username, &config.password) {
            db.signin(surrealdb::opt::auth::Root { username, password })
                .await
                .map_err(|e| AgentError::MemoryError(format!("Failed to authenticate: {}", e)))?;
        }

        // Select namespace and database
        db.use_ns(&config.namespace)
            .use_db(&config.database)
            .await
            .map_err(|e| {
                AgentError::MemoryError(format!("Failed to select namespace/database: {}", e))
            })?;

        // Create table and indexes if they don't exist
        let create_table_query = format!(
            r#"
            DEFINE TABLE {} SCHEMAFULL;
            DEFINE FIELD input ON TABLE {} TYPE string;
            DEFINE FIELD output ON TABLE {} TYPE string;
            DEFINE FIELD embedding ON TABLE {} TYPE array;
            DEFINE FIELD metadata ON TABLE {} TYPE object;
            DEFINE FIELD created_at ON TABLE {} TYPE datetime DEFAULT time::now();
            DEFINE INDEX idx_created_at ON TABLE {} COLUMNS created_at;
            "#,
            config.table,
            config.table,
            config.table,
            config.table,
            config.table,
            config.table,
            config.table
        );

        db.query(&create_table_query)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to create table: {}", e)))?;

        Ok(Self {
            db,
            config,
            embedding_provider,
        })
    }

    /// Find similar memories using vector similarity search
    async fn find_similar(
        &self,
        embedding: &[f32],
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<MemoryRecord>, AgentError> {
        // SurrealDB doesn't have built-in vector similarity yet, so we'll fetch all and compute in-memory
        // In production, you'd want to use a vector database or add vector search to SurrealDB

        let query = format!(
            "SELECT * FROM {} ORDER BY created_at DESC LIMIT 1000",
            self.config.table
        );

        let mut response = self
            .db
            .query(&query)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to query memories: {}", e)))?;

        let records: Vec<MemoryRecord> = response
            .take(0)
            .map_err(|e| AgentError::MemoryError(format!("Failed to parse records: {}", e)))?;

        // Calculate similarities and filter
        let query_embedding = Embedding {
            vector: embedding.to_vec(),
            index: 0,
        };

        let mut similarities: Vec<(f32, MemoryRecord)> = records
            .into_iter()
            .map(|record| {
                let record_embedding = Embedding {
                    vector: record.embedding.clone(),
                    index: 0,
                };
                let similarity = query_embedding.cosine_similarity(&record_embedding);
                (similarity, record)
            })
            .filter(|(similarity, _)| *similarity >= threshold)
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top N
        let results = similarities
            .into_iter()
            .take(limit)
            .map(|(_, record)| record)
            .collect();

        Ok(results)
    }
}

#[async_trait]
impl Memory for SurrealMemoryStore {
    async fn store(&mut self, input: &str, output: &str) -> Result<(), AgentError> {
        // Generate embedding for the input
        let embedding = self
            .embedding_provider
            .embed_single(input)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to generate embedding: {}", e)))?;

        let record = MemoryRecord {
            id: None,
            input: input.to_string(),
            output: output.to_string(),
            embedding: embedding.vector,
            metadata: None,
            created_at: Datetime::default(),
        };

        let query = format!("CREATE {} CONTENT $content", self.config.table);

        self.db
            .query(&query)
            .bind(("content", record))
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to store memory: {}", e)))?;

        Ok(())
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>, AgentError> {
        // Generate embedding for the query
        let embedding = self
            .embedding_provider
            .embed_single(query)
            .await
            .map_err(|e| {
                AgentError::MemoryError(format!("Failed to generate query embedding: {}", e))
            })?;

        // Find similar memories
        let similar_memories = self.find_similar(&embedding.vector, limit, 0.7).await?;

        // Format results
        let results = similar_memories
            .into_iter()
            .map(|record| format!("User: {}\nAssistant: {}", record.input, record.output))
            .collect();

        Ok(results)
    }

    async fn clear(&mut self) -> Result<(), AgentError> {
        let query = format!("DELETE {}", self.config.table);

        self.db
            .query(&query)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to clear memories: {}", e)))?;

        Ok(())
    }

    async fn stats(&self) -> Result<MemoryStats, AgentError> {
        let count_query = format!("SELECT count() FROM {} GROUP ALL", self.config.table);

        let mut response = self
            .db
            .query(&count_query)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to get stats: {}", e)))?;

        let count_result: Option<serde_json::Value> = response
            .take(0)
            .map_err(|e| AgentError::MemoryError(format!("Failed to parse count: {}", e)))?;

        let total_entries = count_result
            .as_ref()
            .and_then(|v| v.get("count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Estimate size (rough approximation)
        let avg_entry_size = 500; // bytes
        let total_size_bytes = total_entries * avg_entry_size;

        Ok(MemoryStats {
            total_entries,
            total_size_bytes,
        })
    }
}

/// Builder for SurrealMemoryStore
#[allow(dead_code)]
pub struct SurrealMemoryBuilder {
    config: SurrealMemoryConfig,
    embedding_provider: Option<Box<dyn EmbeddingProvider>>,
}

#[allow(dead_code)]
impl SurrealMemoryBuilder {
    pub fn new() -> Self {
        Self {
            config: SurrealMemoryConfig::default(),
            embedding_provider: None,
        }
    }

    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.config.url = url.into();
        self
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.config.namespace = namespace.into();
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.config.database = database.into();
        self
    }

    pub fn table(mut self, table: impl Into<String>) -> Self {
        self.config.table = table.into();
        self
    }

    pub fn credentials(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.config.username = Some(username.into());
        self.config.password = Some(password.into());
        self
    }

    pub fn embedding_provider<E: EmbeddingProvider + 'static>(mut self, provider: E) -> Self {
        self.embedding_provider = Some(Box::new(provider));
        self
    }

    pub async fn build(self) -> Result<SurrealMemoryStore, AgentError> {
        let embedding_provider = self
            .embedding_provider
            .ok_or_else(|| AgentError::ConfigError("Embedding provider is required".to_string()))?;

        SurrealMemoryStore::new(self.config, embedding_provider).await
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    #[ignore] // Requires SurrealDB to be running
    async fn test_surrealdb_memory() {
        // This test requires a running SurrealDB instance
        // Run with: cargo test test_surrealdb_memory -- --ignored
    }
}
