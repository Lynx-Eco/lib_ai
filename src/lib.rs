pub mod traits;
pub mod models;
pub mod providers;
pub mod error;
pub mod agent;
pub mod embeddings;
pub mod observability;

pub use traits::*;
pub use models::*;
pub use error::*;

// Re-export derive macros when the derive feature is enabled
#[cfg(feature = "derive")]
pub use lib_ai_derive::{Structured, AiTool};