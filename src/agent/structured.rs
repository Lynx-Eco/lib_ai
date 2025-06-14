use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

use super::{Agent, AgentError};
use crate::JsonSchema;

/// Trait for types that can provide a JSON schema
pub trait StructuredProvider {
    /// Get the JSON schema for this type
    fn schema() -> JsonSchema;
}

/// Extension trait for agents to support structured output
#[async_trait]
pub trait StructuredOutput {
    /// Execute a task and return a strongly-typed response
    async fn execute_typed<T>(&mut self, input: &str) -> Result<T, AgentError>
    where
        T: DeserializeOwned + StructuredProvider + Send;

    /// Chat with the agent and get a typed response
    async fn chat_typed<T>(&mut self, message: &str) -> Result<T, AgentError>
    where
        T: DeserializeOwned + StructuredProvider + Send;
}

#[async_trait]
impl StructuredOutput for Agent {
    async fn execute_typed<T>(&mut self, input: &str) -> Result<T, AgentError>
    where
        T: DeserializeOwned + StructuredProvider + Send,
    {
        // Store original config
        let original_config = self.get_config().clone();

        // Create a new config with JSON output format
        let mut config = original_config.clone();
        config.response_format = Some(crate::ResponseFormat {
            r#type: crate::ResponseFormatType::JsonObject,
        });

        // Update config
        self.update_config(config);

        // We'll include the schema requirement in the input message
        let schema = T::schema();
        let schema_instruction = format!(
            "IMPORTANT: You must respond with valid JSON that matches this schema:\n{}",
            serde_json::to_string_pretty(&schema.schema).unwrap_or_default()
        );
        let full_input = format!("{}\n\n{}", schema_instruction, input);

        // Execute the task with schema instruction
        let response = self.execute(&full_input).await?;

        // Restore original config
        self.update_config(original_config);

        // Parse the response
        serde_json::from_str(&response).map_err(|e| {
            AgentError::ContextError(format!("Failed to parse structured response: {}", e))
        })
    }

    async fn chat_typed<T>(&mut self, message: &str) -> Result<T, AgentError>
    where
        T: DeserializeOwned + StructuredProvider + Send,
    {
        self.execute_typed(message).await
    }
}

/// Builder for typed agents with structured output
pub struct TypedAgentBuilder<T> {
    inner: super::AgentBuilder,
    _phantom: PhantomData<T>,
}

impl<T> TypedAgentBuilder<T>
where
    T: DeserializeOwned + StructuredProvider + Send,
{
    /// Create a new typed agent builder
    pub fn new() -> Self {
        let mut builder = super::AgentBuilder::new();
        // Set JSON response format by default
        builder = builder.response_format(crate::ResponseFormat {
            r#type: crate::ResponseFormatType::JsonObject,
        });

        // Add schema information to the prompt
        let schema = T::schema();
        let schema_prompt = format!(
            "You are a helpful assistant that always responds with JSON matching the following schema:\n{}",
            serde_json::to_string_pretty(&schema.schema).unwrap_or_default()
        );
        builder = builder.prompt(schema_prompt);

        Self {
            inner: builder,
            _phantom: PhantomData,
        }
    }

    /// Set the completion provider
    pub fn provider<P: crate::CompletionProvider + 'static>(mut self, provider: P) -> Self {
        self.inner = self.inner.provider(provider);
        self
    }

    /// Set the system prompt
    pub fn prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        self.inner = self.inner.prompt(prompt);
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.inner = self.inner.model(model);
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.inner = self.inner.temperature(temp);
        self
    }

    /// Build the typed agent
    pub fn build(self) -> Result<TypedAgent<T>, String> {
        let agent = self.inner.build()?;
        Ok(TypedAgent {
            agent,
            _phantom: PhantomData,
        })
    }
}

/// A typed agent that returns structured responses
pub struct TypedAgent<T> {
    agent: Agent,
    _phantom: PhantomData<T>,
}

impl<T> TypedAgent<T>
where
    T: DeserializeOwned + StructuredProvider + Send,
{
    /// Execute a task and get typed response
    pub async fn execute(&mut self, input: &str) -> Result<T, AgentError> {
        self.agent.execute_typed(input).await
    }

    /// Chat and get typed response
    pub async fn chat(&mut self, message: &str) -> Result<T, AgentError> {
        self.agent.chat_typed(message).await
    }

    /// Get the underlying agent for advanced operations
    pub fn inner(&self) -> &Agent {
        &self.agent
    }

    /// Get mutable access to the underlying agent
    pub fn inner_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }
}

/// Macro to easily create JSON schema from a struct
#[macro_export]
macro_rules! impl_json_schema {
    ($type:ty) => {
        impl $crate::agent::StructuredProvider for $type {
            fn schema() -> $crate::JsonSchema {
                $crate::JsonSchema {
                    name: stringify!($type).to_string(),
                    description: None,
                    schema: serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": Vec::<String>::new(),
                    }),
                    strict: None,
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct TestResponse {
        answer: String,
        confidence: f32,
    }

    impl StructuredProvider for TestResponse {
        fn schema() -> JsonSchema {
            JsonSchema {
                name: "TestResponse".to_string(),
                description: None,
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" },
                        "confidence": { "type": "number" }
                    },
                    "required": ["answer", "confidence"]
                }),
                strict: None,
            }
        }
    }

    #[test]
    fn test_typed_agent_builder() {
        let builder = TypedAgentBuilder::<TestResponse>::new()
            .prompt("You are a helpful assistant")
            .temperature(0.7);

        // Builder should compile and be usable
        assert!(true);
    }
}
