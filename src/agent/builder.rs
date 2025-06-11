use std::sync::Arc;

use crate::{
    CompletionProvider,
    observability::{MetricsCollector, AgentTracer, CostTracker, TelemetryExporter},
};
use super::{ Agent, Context, Memory, ToolRegistry, ToolExecutor };
use super::agent::AgentConfig;

/// Builder for creating an Agent with a fluent API
pub struct AgentBuilder {
    provider: Option<Arc<dyn CompletionProvider>>,
    prompt: Option<String>,
    context: Context,
    memory: Option<Box<dyn Memory>>,
    tools: Option<ToolRegistry>,
    config: AgentConfig,
    metrics_collector: Option<Arc<MetricsCollector>>,
    tracer: Option<Arc<AgentTracer>>,
    cost_tracker: Option<Arc<std::sync::RwLock<CostTracker>>>,
    telemetry_exporter: Option<Arc<TelemetryExporter>>,
}

impl AgentBuilder {
    /// Create a new agent builder
    pub fn new() -> Self {
        Self {
            provider: None,
            prompt: None,
            context: Context::new(),
            memory: None,
            tools: None,
            config: AgentConfig::default(),
            metrics_collector: None,
            tracer: None,
            cost_tracker: None,
            telemetry_exporter: None,
        }
    }

    /// Set the completion provider
    pub fn provider<P: CompletionProvider + 'static>(mut self, provider: P) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Set the completion provider (Arc version)
    pub fn provider_arc(mut self, provider: Arc<dyn CompletionProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the system prompt
    pub fn prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        let prompt_str = prompt.into();
        self.context.add_system_message(&prompt_str);
        self.prompt = Some(prompt_str.clone());
        self
    }

    /// Add a preamble message (additional system context)
    pub fn preamble<S: Into<String>>(mut self, preamble: S) -> Self {
        self.context.add_system_message(&preamble.into());
        self
    }

    /// Set the model to use
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config.model = Some(model.into());
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p value
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = Some(top_p);
        self
    }
    
    /// Set the response format
    pub fn response_format(mut self, format: crate::ResponseFormat) -> Self {
        self.config.response_format = Some(format);
        self
    }

    /// Set the maximum iterations for tool use
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Enable streaming
    pub fn stream(mut self, stream: bool) -> Self {
        self.config.stream = stream;
        self
    }

    /// Add memory to the agent
    pub fn memory<M: Memory + 'static>(mut self, memory: M) -> Self {
        self.memory = Some(Box::new(memory));
        self
    }

    /// Add a single tool
    pub fn tool<S, E>(mut self, name: S, executor: E) -> Self
        where S: Into<String>, E: ToolExecutor + 'static
    {
        if self.tools.is_none() {
            self.tools = Some(ToolRegistry::new());
        }

        if let Some(tools) = &mut self.tools {
            tools.register(name, executor);
        }

        self
    }

    /// Add multiple tools from a registry
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Add initial context messages
    pub fn context(mut self, context: Context) -> Self {
        // Preserve system messages
        let system_messages = self.context
            .messages()
            .filter(|m| matches!(m.role, crate::Role::System))
            .cloned()
            .collect::<Vec<_>>();

        self.context = context;

        // Re-add system messages at the beginning
        for (i, msg) in system_messages.into_iter().enumerate() {
            self.context.messages_mut().insert(i, super::context::ContextMessage {
                message: msg,
                timestamp: std::time::SystemTime::now(),
                metadata: None,
            });
        }

        self
    }

    /// Add a context message
    pub fn add_message(mut self, role: crate::Role, content: &str) -> Self {
        use crate::MessageContent;

        self.context.add_message(crate::Message {
            role,
            content: MessageContent::text(content),
            tool_calls: None,
            tool_call_id: None,
        });
        self
    }

    /// Set metrics collector for observability
    pub fn metrics_collector(mut self, metrics_collector: Arc<MetricsCollector>) -> Self {
        self.metrics_collector = Some(metrics_collector);
        self
    }

    /// Set tracer for distributed tracing
    pub fn tracer(mut self, tracer: Arc<AgentTracer>) -> Self {
        self.tracer = Some(tracer);
        self
    }

    /// Set cost tracker for cost monitoring
    pub fn cost_tracker(mut self, cost_tracker: Arc<std::sync::RwLock<CostTracker>>) -> Self {
        self.cost_tracker = Some(cost_tracker);
        self
    }

    /// Set telemetry exporter for data export
    pub fn telemetry_exporter(mut self, telemetry_exporter: Arc<TelemetryExporter>) -> Self {
        self.telemetry_exporter = Some(telemetry_exporter);
        self
    }

    /// Enable full observability with all components
    pub fn with_observability(
        mut self,
        metrics_collector: Arc<MetricsCollector>,
        tracer: Arc<AgentTracer>,
        cost_tracker: Arc<std::sync::RwLock<CostTracker>>,
        telemetry_exporter: Arc<TelemetryExporter>,
    ) -> Self {
        self.metrics_collector = Some(metrics_collector);
        self.tracer = Some(tracer);
        self.cost_tracker = Some(cost_tracker);
        self.telemetry_exporter = Some(telemetry_exporter);
        self
    }

    /// Build the agent
    pub fn build(self) -> Result<Agent, String> {
        let provider = self.provider.ok_or_else(|| "Provider is required".to_string())?;

        let prompt = self.prompt.unwrap_or_default();

        let agent = Agent::new(provider, prompt, self.context, self.memory, self.tools, self.config)
            .with_observability(
                self.metrics_collector,
                self.tracer,
                self.cost_tracker,
                self.telemetry_exporter,
            );

        Ok(agent)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        // This would need a mock provider for real testing
        let builder = AgentBuilder::new()
            .prompt("You are a helpful assistant")
            .model("gpt-4")
            .temperature(0.7)
            .max_tokens(1000);

        assert_eq!(builder.config.temperature, Some(0.7));
        assert_eq!(builder.config.max_tokens, Some(1000));
    }
}
