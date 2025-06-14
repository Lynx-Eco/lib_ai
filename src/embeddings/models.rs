use serde::{Deserialize, Serialize};

/// A single embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,

    /// The index of this embedding in the request
    pub index: usize,
}

/// Request for generating embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// The texts to embed
    pub input: Vec<String>,

    /// The model to use
    pub model: String,
}

/// Response containing embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The generated embeddings
    pub embeddings: Vec<Embedding>,

    /// Token usage information
    pub usage: Option<EmbeddingUsage>,
}

/// Token usage for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl Embedding {
    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();

        let magnitude_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        let magnitude_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }

    /// Calculate Euclidean distance between two embeddings
    pub fn euclidean_distance(&self, other: &Embedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return f32::MAX;
        }

        self.vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}
