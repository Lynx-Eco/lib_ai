use async_trait::async_trait;
use std::sync::{Arc, Mutex};

use crate::agent::AgentError;

/// Trait for agent memory storage
#[async_trait]
pub trait Memory: Send + Sync {
    /// Store a conversation turn in memory
    async fn store(&mut self, input: &str, output: &str) -> Result<(), AgentError>;
    
    /// Retrieve relevant memories based on a query
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>, AgentError>;
    
    /// Clear all memories
    async fn clear(&mut self) -> Result<(), AgentError>;
    
    /// Get memory statistics
    async fn stats(&self) -> Result<MemoryStats, AgentError>;
}

/// Statistics about the memory store
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
}

/// Simple in-memory storage implementation
#[derive(Clone)]
pub struct InMemoryStore {
    entries: Arc<Mutex<Vec<MemoryEntry>>>,
    max_entries: usize,
}

#[derive(Clone)]
struct MemoryEntry {
    input: String,
    output: String,
    #[allow(dead_code)]
    timestamp: std::time::SystemTime,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            max_entries,
        }
    }
}

#[async_trait]
impl Memory for InMemoryStore {
    async fn store(&mut self, input: &str, output: &str) -> Result<(), AgentError> {
        let mut entries = self.entries.lock().unwrap();
        
        entries.push(MemoryEntry {
            input: input.to_string(),
            output: output.to_string(),
            timestamp: std::time::SystemTime::now(),
        });
        
        // Enforce max entries limit
        if entries.len() > self.max_entries {
            entries.remove(0);
        }
        
        Ok(())
    }
    
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>, AgentError> {
        let entries = self.entries.lock().unwrap();
        
        // Simple similarity: find entries where input contains query words
        let query_words: Vec<&str> = query.split_whitespace().collect();
        
        let mut matches: Vec<(usize, &MemoryEntry)> = entries
            .iter()
            .enumerate()
            .filter_map(|(_idx, entry)| {
                let score = query_words.iter()
                    .filter(|word| entry.input.to_lowercase().contains(&word.to_lowercase()))
                    .count();
                
                if score > 0 {
                    Some((score, entry))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by relevance (score) descending
        matches.sort_by(|a, b| b.0.cmp(&a.0));
        
        // Take top matches and format
        let results = matches
            .into_iter()
            .take(limit)
            .map(|(_, entry)| {
                format!("User: {}\nAssistant: {}", entry.input, entry.output)
            })
            .collect();
        
        Ok(results)
    }
    
    async fn clear(&mut self) -> Result<(), AgentError> {
        let mut entries = self.entries.lock().unwrap();
        entries.clear();
        Ok(())
    }
    
    async fn stats(&self) -> Result<MemoryStats, AgentError> {
        let entries = self.entries.lock().unwrap();
        
        let total_size_bytes: usize = entries
            .iter()
            .map(|e| e.input.len() + e.output.len())
            .sum();
        
        Ok(MemoryStats {
            total_entries: entries.len(),
            total_size_bytes,
        })
    }
}

/// A more sophisticated memory store with semantic search
pub struct SemanticMemoryStore {
    // This would integrate with a vector database
    // For now, we'll use the in-memory store as a base
    base: InMemoryStore,
}

impl SemanticMemoryStore {
    pub fn new(max_entries: usize) -> Self {
        Self {
            base: InMemoryStore::new(max_entries),
        }
    }
}

#[async_trait]
impl Memory for SemanticMemoryStore {
    async fn store(&mut self, input: &str, output: &str) -> Result<(), AgentError> {
        // In a real implementation, this would:
        // 1. Generate embeddings for input/output
        // 2. Store in vector database
        self.base.store(input, output).await
    }
    
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>, AgentError> {
        // In a real implementation, this would:
        // 1. Generate embedding for query
        // 2. Perform semantic search in vector database
        self.base.retrieve(query, limit).await
    }
    
    async fn clear(&mut self) -> Result<(), AgentError> {
        self.base.clear().await
    }
    
    async fn stats(&self) -> Result<MemoryStats, AgentError> {
        self.base.stats().await
    }
}

/// Memory store that persists to disk
pub struct PersistentMemoryStore {
    base: InMemoryStore,
    file_path: std::path::PathBuf,
}

impl PersistentMemoryStore {
    pub fn new(file_path: std::path::PathBuf, max_entries: usize) -> Result<Self, AgentError> {
        let mut store = Self {
            base: InMemoryStore::new(max_entries),
            file_path,
        };
        
        // Load existing data if file exists
        if store.file_path.exists() {
            store.load_from_disk()?;
        }
        
        Ok(store)
    }
    
    fn load_from_disk(&mut self) -> Result<(), AgentError> {
        use std::fs;
        
        let content = fs::read_to_string(&self.file_path)
            .map_err(|e| AgentError::MemoryError(format!("Failed to read memory file: {}", e)))?;
        
        let entries: Vec<(String, String)> = serde_json::from_str(&content)
            .map_err(|e| AgentError::MemoryError(format!("Failed to parse memory file: {}", e)))?;
        
        let base_clone = self.base.clone();
        let rt = tokio::runtime::Handle::current();
        for (input, output) in entries {
            rt.block_on(async {
                let mut base = base_clone.clone();
                base.store(&input, &output).await
            })?;
        }
        
        Ok(())
    }
    
    fn save_to_disk(&self) -> Result<(), AgentError> {
        use std::fs;
        
        let entries = self.base.entries.lock().unwrap();
        let data: Vec<(&str, &str)> = entries
            .iter()
            .map(|e| (e.input.as_str(), e.output.as_str()))
            .collect();
        
        let content = serde_json::to_string_pretty(&data)
            .map_err(|e| AgentError::MemoryError(format!("Failed to serialize memory: {}", e)))?;
        
        fs::write(&self.file_path, content)
            .map_err(|e| AgentError::MemoryError(format!("Failed to write memory file: {}", e)))?;
        
        Ok(())
    }
}

#[async_trait]
impl Memory for PersistentMemoryStore {
    async fn store(&mut self, input: &str, output: &str) -> Result<(), AgentError> {
        self.base.store(input, output).await?;
        self.save_to_disk()?;
        Ok(())
    }
    
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>, AgentError> {
        self.base.retrieve(query, limit).await
    }
    
    async fn clear(&mut self) -> Result<(), AgentError> {
        self.base.clear().await?;
        self.save_to_disk()?;
        Ok(())
    }
    
    async fn stats(&self) -> Result<MemoryStats, AgentError> {
        self.base.stats().await
    }
}

/// A trait for implementing custom memory stores
pub trait MemoryStore: Memory {
    /// Create a new instance of the memory store
    fn new() -> Self;
    
    /// Get the name of the memory store
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_in_memory_store() {
        let mut store = InMemoryStore::new(10);
        
        // Store some conversations
        store.store("What's the weather?", "I don't have access to weather data.").await.unwrap();
        store.store("Tell me a joke", "Why did the chicken cross the road?").await.unwrap();
        
        // Retrieve relevant memories
        let results = store.retrieve("weather", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("weather"));
        
        // Test stats
        let stats = store.stats().await.unwrap();
        assert_eq!(stats.total_entries, 2);
    }
}