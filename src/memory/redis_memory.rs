use crate::memory::{Memory, MemoryBackend, MemoryEntry, MemoryQuery, MemoryQueryBuilder};
use crate::error::{AiError, Result};
use async_trait::async_trait;
use redis::{aio::MultiplexedConnection, AsyncCommands, Client, RedisResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct RedisMemory {
    client: Client,
    connection: MultiplexedConnection,
    namespace: String,
    ttl_seconds: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SerializedMemoryEntry {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
    pub embedding: Option<Vec<f32>>,
}

impl From<MemoryEntry> for SerializedMemoryEntry {
    fn from(entry: MemoryEntry) -> Self {
        Self {
            id: entry.id,
            role: entry.role.to_string(),
            content: entry.content,
            timestamp: entry.timestamp,
            metadata: entry.metadata,
            embedding: entry.embedding,
        }
    }
}

impl RedisMemory {
    pub async fn new(redis_url: &str, namespace: &str) -> Result<Self> {
        let client = Client::open(redis_url)
            .map_err(|e| AiError::MemoryError(format!("Failed to create Redis client: {}", e)))?;
        
        let connection = client.get_multiplexed_async_connection().await
            .map_err(|e| AiError::MemoryError(format!("Failed to connect to Redis: {}", e)))?;
        
        Ok(Self {
            client,
            connection,
            namespace: namespace.to_string(),
            ttl_seconds: None,
        })
    }

    pub fn with_ttl(mut self, ttl_seconds: i64) -> Self {
        self.ttl_seconds = Some(ttl_seconds);
        self
    }

    fn make_key(&self, key: &str) -> String {
        format!("{}:{}", self.namespace, key)
    }

    fn make_list_key(&self) -> String {
        format!("{}:entries", self.namespace)
    }

    fn make_metadata_key(&self, metadata_key: &str, metadata_value: &str) -> String {
        format!("{}:metadata:{}:{}", self.namespace, metadata_key, metadata_value)
    }

    fn make_role_key(&self, role: &str) -> String {
        format!("{}:role:{}", self.namespace, role)
    }

    async fn add_to_indices(&mut self, entry: &SerializedMemoryEntry) -> Result<()> {
        // Add to main list
        let _: () = self.connection.lpush(self.make_list_key(), &entry.id).await
            .map_err(|e| AiError::MemoryError(format!("Failed to add to list index: {}", e)))?;

        // Add to role index
        let _: () = self.connection.sadd(self.make_role_key(&entry.role), &entry.id).await
            .map_err(|e| AiError::MemoryError(format!("Failed to add to role index: {}", e)))?;

        // Add to metadata indices
        for (key, value) in &entry.metadata {
            let _: () = self.connection.sadd(self.make_metadata_key(key, value), &entry.id).await
                .map_err(|e| AiError::MemoryError(format!("Failed to add to metadata index: {}", e)))?;
        }

        Ok(())
    }

    async fn remove_from_indices(&mut self, entry: &SerializedMemoryEntry) -> Result<()> {
        // Remove from main list
        let _: () = self.connection.lrem(self.make_list_key(), 0, &entry.id).await
            .map_err(|e| AiError::MemoryError(format!("Failed to remove from list index: {}", e)))?;

        // Remove from role index
        let _: () = self.connection.srem(self.make_role_key(&entry.role), &entry.id).await
            .map_err(|e| AiError::MemoryError(format!("Failed to remove from role index: {}", e)))?;

        // Remove from metadata indices
        for (key, value) in &entry.metadata {
            let _: () = self.connection.srem(self.make_metadata_key(key, value), &entry.id).await
                .map_err(|e| AiError::MemoryError(format!("Failed to remove from metadata index: {}", e)))?;
        }

        Ok(())
    }
}

#[async_trait]
impl MemoryBackend for RedisMemory {
    async fn add(&mut self, entry: MemoryEntry) -> Result<()> {
        let serialized = SerializedMemoryEntry::from(entry);
        let key = self.make_key(&serialized.id);
        let value = serde_json::to_string(&serialized)
            .map_err(|e| AiError::MemoryError(format!("Failed to serialize entry: {}", e)))?;

        // Store the entry
        if let Some(ttl) = self.ttl_seconds {
            let _: () = self.connection.set_ex(&key, value, ttl as u64).await
                .map_err(|e| AiError::MemoryError(format!("Failed to store entry: {}", e)))?;
        } else {
            let _: () = self.connection.set(&key, value).await
                .map_err(|e| AiError::MemoryError(format!("Failed to store entry: {}", e)))?;
        }

        // Update indices
        self.add_to_indices(&serialized).await?;

        Ok(())
    }

    async fn get(&mut self, id: &str) -> Result<Option<MemoryEntry>> {
        let key = self.make_key(id);
        let value: Option<String> = self.connection.get(&key).await
            .map_err(|e| AiError::MemoryError(format!("Failed to get entry: {}", e)))?;

        match value {
            Some(json) => {
                let serialized: SerializedMemoryEntry = serde_json::from_str(&json)
                    .map_err(|e| AiError::MemoryError(format!("Failed to deserialize entry: {}", e)))?;
                
                Ok(Some(MemoryEntry {
                    id: serialized.id,
                    role: serialized.role.parse()
                        .map_err(|_| AiError::MemoryError("Invalid role in stored entry".to_string()))?,
                    content: serialized.content,
                    timestamp: serialized.timestamp,
                    metadata: serialized.metadata,
                    embedding: serialized.embedding,
                }))
            }
            None => Ok(None),
        }
    }

    async fn update(&mut self, id: &str, entry: MemoryEntry) -> Result<()> {
        // Get the old entry to update indices
        if let Some(old_entry) = self.get(id).await? {
            let old_serialized = SerializedMemoryEntry::from(old_entry);
            self.remove_from_indices(&old_serialized).await?;
        }

        // Store the new entry
        self.add(entry).await
    }

    async fn delete(&mut self, id: &str) -> Result<()> {
        // Get the entry to update indices
        if let Some(entry) = self.get(id).await? {
            let serialized = SerializedMemoryEntry::from(entry);
            self.remove_from_indices(&serialized).await?;
        }

        let key = self.make_key(id);
        let _: () = self.connection.del(&key).await
            .map_err(|e| AiError::MemoryError(format!("Failed to delete entry: {}", e)))?;

        Ok(())
    }

    async fn query(&mut self, query: MemoryQuery) -> Result<Vec<MemoryEntry>> {
        let mut entry_ids: Vec<String> = Vec::new();

        // If role filter is specified, use role index
        if let Some(role) = &query.role {
            let role_ids: Vec<String> = self.connection.smembers(self.make_role_key(&role.to_string())).await
                .map_err(|e| AiError::MemoryError(format!("Failed to query role index: {}", e)))?;
            entry_ids = role_ids;
        } else {
            // Get all entries from the list
            let all_ids: Vec<String> = self.connection.lrange(self.make_list_key(), 0, -1).await
                .map_err(|e| AiError::MemoryError(format!("Failed to query list index: {}", e)))?;
            entry_ids = all_ids;
        }

        // Filter by metadata if specified
        for (key, value) in &query.metadata {
            let metadata_ids: Vec<String> = self.connection.smembers(self.make_metadata_key(key, value)).await
                .map_err(|e| AiError::MemoryError(format!("Failed to query metadata index: {}", e)))?;
            
            // Intersect with existing ids
            entry_ids.retain(|id| metadata_ids.contains(id));
        }

        // Fetch all matching entries
        let mut entries = Vec::new();
        for id in entry_ids {
            if let Some(entry) = self.get(&id).await? {
                // Apply time filters
                if let Some(start) = query.start_time {
                    if entry.timestamp < start {
                        continue;
                    }
                }
                if let Some(end) = query.end_time {
                    if entry.timestamp > end {
                        continue;
                    }
                }

                entries.push(entry);
            }
        }

        // Sort by timestamp
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Apply limit
        if let Some(limit) = query.limit {
            entries.truncate(limit);
        }

        Ok(entries)
    }

    async fn clear(&mut self) -> Result<()> {
        // Get all keys with our namespace
        let pattern = format!("{}:*", self.namespace);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut self.connection)
            .await
            .map_err(|e| AiError::MemoryError(format!("Failed to get keys: {}", e)))?;

        if !keys.is_empty() {
            let _: () = self.connection.del(keys).await
                .map_err(|e| AiError::MemoryError(format!("Failed to clear entries: {}", e)))?;
        }

        Ok(())
    }

    async fn count(&mut self) -> Result<usize> {
        let count: isize = self.connection.llen(self.make_list_key()).await
            .map_err(|e| AiError::MemoryError(format!("Failed to count entries: {}", e)))?;
        
        Ok(count as usize)
    }
}

impl Memory<RedisMemory> {
    pub async fn with_redis(redis_url: &str, namespace: &str) -> Result<Self> {
        let backend = RedisMemory::new(redis_url, namespace).await?;
        Ok(Self::new(Box::new(backend)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Role;

    #[tokio::test]
    async fn test_redis_memory_basic_operations() -> Result<()> {
        // This test requires a Redis instance running
        // Skip if Redis is not available
        let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        
        let mut memory = match Memory::with_redis(&redis_url, "test_namespace").await {
            Ok(m) => m,
            Err(_) => {
                eprintln!("Redis not available, skipping test");
                return Ok(());
            }
        };

        // Clear any existing data
        memory.clear().await?;

        // Test add
        let entry = MemoryEntry {
            id: Uuid::new_v4().to_string(),
            role: Role::User,
            content: "Hello, Redis!".to_string(),
            timestamp: Utc::now(),
            metadata: HashMap::from([("tag".to_string(), "greeting".to_string())]),
            embedding: None,
        };

        memory.add(entry.clone()).await?;

        // Test get
        let retrieved = memory.get(&entry.id).await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Hello, Redis!");

        // Test count
        assert_eq!(memory.count().await?, 1);

        // Test query
        let query = MemoryQueryBuilder::new()
            .with_role(Role::User)
            .with_metadata("tag", "greeting")
            .build();
        
        let results = memory.query(query).await?;
        assert_eq!(results.len(), 1);

        // Test delete
        memory.delete(&entry.id).await?;
        assert_eq!(memory.count().await?, 0);

        Ok(())
    }
}