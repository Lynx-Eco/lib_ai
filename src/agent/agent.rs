use std::sync::Arc;
use thiserror::Error;

use crate::{
    CompletionProvider, CompletionRequest, CompletionResponse, 
    ToolCall, ToolChoice, ResponseFormat,
};
use super::{Context, Memory, ToolRegistry, ToolResult};

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Provider error: {0}")]
    ProviderError(#[from] crate::AiError),
    
    #[error("Tool execution error: {0}")]
    ToolError(String),
    
    #[error("Context error: {0}")]
    ContextError(String),
    
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, AgentError>;

/// Configuration for an agent
#[derive(Clone)]
pub struct AgentConfig {
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub response_format: Option<ResponseFormat>,
    pub max_iterations: usize,
    pub stream: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            response_format: None,
            max_iterations: 10,
            stream: false,
        }
    }
}

/// An AI agent that can complete tasks using tools and memory
pub struct Agent {
    provider: Arc<dyn CompletionProvider>,
    #[allow(dead_code)]
    prompt: String,
    context: Context,
    memory: Option<Box<dyn Memory>>,
    tools: Option<ToolRegistry>,
    config: AgentConfig,
}

impl Agent {
    pub(crate) fn new(
        provider: Arc<dyn CompletionProvider>,
        prompt: String,
        context: Context,
        memory: Option<Box<dyn Memory>>,
        tools: Option<ToolRegistry>,
        config: AgentConfig,
    ) -> Self {
        Self {
            provider,
            prompt,
            context,
            memory,
            tools,
            config,
        }
    }

    /// Execute a task with the given input
    pub async fn execute(&mut self, input: &str) -> Result<String> {
        // Add user input to context
        self.context.add_user_message(input);
        
        // Retrieve relevant memory if available
        if let Some(memory) = &self.memory {
            let memories = memory.retrieve(input, 5).await?;
            for mem in memories {
                self.context.add_memory(mem);
            }
        }
        
        // Main execution loop
        let mut iterations = 0;
        let final_response;
        
        loop {
            if iterations >= self.config.max_iterations {
                return Err(AgentError::ConfigError(
                    format!("Maximum iterations ({}) reached", self.config.max_iterations)
                ));
            }
            
            // Build the completion request
            let request = self.build_request()?;
            
            // Get completion from provider
            let response = self.provider.complete(request).await?;
            
            // Process the response
            let (should_continue, response_text) = self.process_response(response).await?;
            
            if !should_continue {
                final_response = response_text;
                break;
            }
            
            iterations += 1;
        }
        
        // Store interaction in memory if available
        if let Some(memory) = &mut self.memory {
            memory.store(input, &final_response).await?;
        }
        
        Ok(final_response)
    }

    /// Execute with streaming response
    pub async fn execute_stream(
        &mut self,
        input: &str,
    ) -> Result<impl futures::Stream<Item = Result<String>>> {
        use futures::stream::StreamExt;
        
        // Add user input to context
        self.context.add_user_message(input);
        
        // Build the completion request
        let mut request = self.build_request()?;
        request.stream = Some(true);
        
        // Get streaming completion from provider
        let stream = self.provider.complete_stream(request).await?;
        
        // Transform the stream
        let transformed_stream = stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let mut content = String::new();
                    for choice in chunk.choices {
                        if let Some(delta_content) = choice.delta.content {
                            content.push_str(&delta_content);
                        }
                    }
                    Ok(content)
                }
                Err(e) => Err(AgentError::ProviderError(e)),
            }
        });
        
        Ok(transformed_stream)
    }

    /// Chat with the agent (maintains conversation context)
    pub async fn chat(&mut self, message: &str) -> Result<String> {
        self.execute(message).await
    }

    /// Clear the conversation context
    pub fn clear_context(&mut self) {
        self.context.clear();
    }

    /// Get the current context
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &AgentConfig {
        &self.config
    }
    
    /// Update the agent's configuration
    pub fn update_config(&mut self, config: AgentConfig) {
        self.config = config;
    }

    fn build_request(&self) -> Result<CompletionRequest> {
        let messages = self.context.to_messages();
        
        let model = self.config.model.clone()
            .unwrap_or_else(|| self.provider.default_model().to_string());
        
        let tools = self.tools.as_ref().map(|registry| registry.to_tools());
        let tool_choice = if tools.is_some() {
            Some(ToolChoice::String("auto".to_string()))
        } else {
            None
        };
        
        Ok(CompletionRequest {
            model,
            messages,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            stream: Some(self.config.stream),
            top_p: self.config.top_p,
            tools,
            tool_choice,
            response_format: self.config.response_format.clone(),
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            json_schema: None,
        })
    }

    async fn process_response(&mut self, response: CompletionResponse) -> Result<(bool, String)> {
        if response.choices.is_empty() {
            return Err(AgentError::ProviderError(crate::AiError::InvalidRequest(
                "No choices in response".to_string()
            )));
        }
        
        let choice = &response.choices[0];
        let message = &choice.message;
        
        // Add assistant message to context
        self.context.add_message(message.clone());
        
        // Check if there are tool calls
        if let Some(tool_calls) = &message.tool_calls {
            // Execute tools
            for tool_call in tool_calls {
                let result = self.execute_tool(tool_call).await?;
                
                // Add tool result to context
                self.context.add_tool_result(&tool_call.id, &result);
            }
            
            // Continue conversation after tool execution
            Ok((true, String::new()))
        } else {
            // Extract text content and return
            let text = message.content.as_text()
                .ok_or_else(|| AgentError::ContextError("No text content in response".to_string()))?;
            Ok((false, text.to_string()))
        }
    }

    async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        let tools = self.tools.as_ref()
            .ok_or_else(|| AgentError::ToolError("No tools available".to_string()))?;
        
        let executor = tools.get_executor(&tool_call.function.name)
            .ok_or_else(|| AgentError::ToolError(
                format!("Tool '{}' not found", tool_call.function.name)
            ))?;
        
        let result = executor.execute(&tool_call.function.arguments).await
            .map_err(|e| AgentError::ToolError(e.to_string()))?;
        
        match result {
            ToolResult::Success(value) => Ok(serde_json::to_string(&value).unwrap_or_else(|_| value.to_string())),
            ToolResult::Error(error) => Err(AgentError::ToolError(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    
    #[tokio::test]
    async fn test_agent_creation() {
        // This test verifies the agent can be created
        // Real tests would use a mock provider
    }
}