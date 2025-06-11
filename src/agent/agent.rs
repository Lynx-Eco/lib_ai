use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

use crate::{
    CompletionProvider, CompletionRequest, CompletionResponse, 
    ToolCall, ToolChoice, ResponseFormat,
    observability::{MetricsCollector, AgentTracer, CostTracker, TelemetryExporter},
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
    agent_id: String,
    metrics_collector: Option<Arc<MetricsCollector>>,
    tracer: Option<Arc<AgentTracer>>,
    cost_tracker: Option<Arc<std::sync::RwLock<CostTracker>>>,
    telemetry_exporter: Option<Arc<TelemetryExporter>>,
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
        let agent_id = uuid::Uuid::new_v4().to_string();
        Self {
            provider,
            prompt,
            context,
            memory,
            tools,
            config,
            agent_id,
            metrics_collector: None,
            tracer: None,
            cost_tracker: None,
            telemetry_exporter: None,
        }
    }

    /// Set observability components
    pub fn with_observability(
        mut self,
        metrics_collector: Option<Arc<MetricsCollector>>,
        tracer: Option<Arc<AgentTracer>>,
        cost_tracker: Option<Arc<std::sync::RwLock<CostTracker>>>,
        telemetry_exporter: Option<Arc<TelemetryExporter>>,
    ) -> Self {
        self.metrics_collector = metrics_collector;
        self.tracer = tracer;
        self.cost_tracker = cost_tracker;
        self.telemetry_exporter = telemetry_exporter;
        self
    }

    /// Get the agent ID
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Execute a task with the given input
    pub async fn execute(&mut self, input: &str) -> Result<String> {
        let start_time = Instant::now();
        let mut total_tokens = crate::observability::metrics::TokenUsage::new();
        let mut total_cost = 0.0;
        
        // Start trace span if tracer is available
        let _trace_span = self.tracer.as_ref().and_then(|tracer| {
            tracer.start_trace(format!("agent_execute_{}", self.agent_id))
        });
        
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
        let mut final_response = String::new();
        
        let execution_result = loop {
            if iterations >= self.config.max_iterations {
                break Err(AgentError::ConfigError(
                    format!("Maximum iterations ({}) reached", self.config.max_iterations)
                ));
            }
            
            // Build the completion request
            let request = self.build_request()?;
            let model = request.model.clone();
            
            // Get completion from provider
            let response = self.provider.complete(request).await?;
            
            // Track tokens and costs
            if let Some(usage) = &response.usage {
                total_tokens.input_tokens += usage.prompt_tokens as u64;
                total_tokens.output_tokens += usage.completion_tokens as u64;
                
                // Calculate cost if cost tracker is available
                if let Some(cost_tracker) = &self.cost_tracker {
                    if let Ok(mut tracker) = cost_tracker.write() {
                        let pricing = tracker.get_pricing(self.provider.name(), &model);
                        let request_cost = pricing.calculate_cost(
                            usage.prompt_tokens as u64,
                            usage.completion_tokens as u64,
                            0, // cache_read_tokens
                            0, // cache_write_tokens
                        );
                        total_cost += request_cost;
                        
                        tracker.record_usage(
                            self.provider.name(),
                            &model,
                            usage.prompt_tokens as u64,
                            usage.completion_tokens as u64,
                            0,
                            0,
                            &pricing,
                        );
                    }
                }
            }
            
            // Process the response
            let (should_continue, response_text) = self.process_response(response).await?;
            
            if !should_continue {
                final_response = response_text;
                break Ok(());
            }
            
            iterations += 1;
        };
        
        let duration = start_time.elapsed();
        let success = execution_result.is_ok();
        
        // Record metrics if metrics collector is available
        if let Some(metrics) = &self.metrics_collector {
            let model = self.config.model.clone()
                .unwrap_or_else(|| self.provider.default_model().to_string());
            metrics.record_request(
                &self.agent_id,
                success,
                duration,
                total_tokens,
                total_cost,
                self.provider.name(),
                &model,
            );
        }
        
        // Handle execution result
        execution_result?;
        
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
            return Err(AgentError::ProviderError(crate::AiError::InvalidRequest { message: 
                "No choices in response".to_string(), field: None, code: None }
            ))
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
        let start_time = Instant::now();
        let tool_name = &tool_call.function.name;
        
        // Start tool trace span if tracer is available
        let _trace_span = self.tracer.as_ref().and_then(|tracer| {
            tracer.start_trace(format!("tool_execute_{}", tool_name))
        });
        
        let tools = self.tools.as_ref()
            .ok_or_else(|| AgentError::ToolError("No tools available".to_string()))?;
        
        let executor = tools.get_executor(tool_name)
            .ok_or_else(|| AgentError::ToolError(
                format!("Tool '{}' not found", tool_name)
            ))?;
        
        let result = executor.execute(&tool_call.function.arguments).await
            .map_err(|e| AgentError::ToolError(e.to_string()))?;
        
        let duration = start_time.elapsed();
        let success = matches!(result, ToolResult::Success(_));
        
        // Record tool metrics if metrics collector is available
        if let Some(metrics) = &self.metrics_collector {
            let error_type = if success { None } else { Some("tool_execution_error".to_string()) };
            metrics.record_tool_execution(&self.agent_id, tool_name, success, duration, error_type);
        }
        
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