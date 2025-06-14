use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::base::{Memory, MemoryStats};
use crate::agent::AgentError;
use crate::embeddings::{Embedding, EmbeddingProvider};

/// Entry in semantic memory with embedding
#[derive(Clone)]
struct SemanticEntry {
    input: String,
    output: String,
    embedding: Embedding,
    #[allow(dead_code)]
    timestamp: std::time::SystemTime,
}

/// Enhanced semantic memory store with vector similarity search
pub struct EnhancedSemanticMemory {
    entries: Arc<Mutex<Vec<SemanticEntry>>>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    max_entries: usize,
    similarity_threshold: f32,
}

impl EnhancedSemanticMemory {
    /// Create a new semantic memory store
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        max_entries: usize,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            embedding_provider,
            max_entries,
            similarity_threshold,
        }
    }

    /// Find the most similar entries
    async fn find_similar(&self, query_embedding: &Embedding, limit: usize) -> Vec<SemanticEntry> {
        let entries = self.entries.lock().await;

        let mut similarities: Vec<(f32, &SemanticEntry)> = entries
            .iter()
            .map(|entry| {
                let similarity = query_embedding.cosine_similarity(&entry.embedding);
                (similarity, entry)
            })
            .filter(|(similarity, _)| *similarity >= self.similarity_threshold)
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top N and clone
        similarities
            .into_iter()
            .take(limit)
            .map(|(_, entry)| entry.clone())
            .collect()
    }
}

#[async_trait]
impl Memory for EnhancedSemanticMemory {
    async fn store(&mut self, input: &str, output: &str) -> Result<(), AgentError> {
        // Generate embedding for the input
        let embedding = self
            .embedding_provider
            .embed_single(input)
            .await
            .map_err(|e| AgentError::MemoryError(format!("Failed to generate embedding: {}", e)))?;

        let entry = SemanticEntry {
            input: input.to_string(),
            output: output.to_string(),
            embedding,
            timestamp: std::time::SystemTime::now(),
        };

        let mut entries = self.entries.lock().await;
        entries.push(entry);

        // Enforce max entries limit
        if entries.len() > self.max_entries {
            entries.remove(0);
        }

        Ok(())
    }

    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>, AgentError> {
        // Generate embedding for the query
        let query_embedding = self
            .embedding_provider
            .embed_single(query)
            .await
            .map_err(|e| {
                AgentError::MemoryError(format!("Failed to generate query embedding: {}", e))
            })?;

        // Find similar entries
        let similar_entries = self.find_similar(&query_embedding, limit).await;

        // Format results
        let results = similar_entries
            .into_iter()
            .map(|entry| format!("User: {}\nAssistant: {}", entry.input, entry.output))
            .collect();

        Ok(results)
    }

    async fn clear(&mut self) -> Result<(), AgentError> {
        let mut entries = self.entries.lock().await;
        entries.clear();
        Ok(())
    }

    async fn stats(&self) -> Result<MemoryStats, AgentError> {
        let entries = self.entries.lock().await;

        let total_size_bytes: usize = entries
            .iter()
            .map(|e| e.input.len() + e.output.len() + (e.embedding.vector.len() * 4))
            .sum();

        Ok(MemoryStats {
            total_entries: entries.len(),
            total_size_bytes,
        })
    }
}

/// Builder for EnhancedSemanticMemory
pub struct SemanticMemoryBuilder {
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    max_entries: usize,
    similarity_threshold: f32,
}

impl SemanticMemoryBuilder {
    pub fn new() -> Self {
        Self {
            embedding_provider: None,
            max_entries: 1000,
            similarity_threshold: 0.7,
        }
    }

    pub fn embedding_provider<E: EmbeddingProvider + 'static>(mut self, provider: E) -> Self {
        self.embedding_provider = Some(Arc::new(provider));
        self
    }

    pub fn embedding_provider_arc(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    pub fn max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> Result<EnhancedSemanticMemory, String> {
        let provider = self
            .embedding_provider
            .ok_or_else(|| "Embedding provider is required".to_string())?;

        Ok(EnhancedSemanticMemory::new(
            provider,
            self.max_entries,
            self.similarity_threshold,
        ))
    }
}

impl Default for SemanticMemoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
