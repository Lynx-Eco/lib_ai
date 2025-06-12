# Agent System Documentation

The lib_ai Agent system provides a high-level abstraction for building AI-powered agents with tool use, memory, and conversation management.

## Overview

Agents in lib_ai are built using a fluent builder pattern and can:
- Use any supported AI provider
- Execute tools/functions
- Maintain conversation context
- Store and retrieve memories
- Stream responses
- Handle multi-turn conversations

## Quick Start

```rust
use lib_ai::{
    agent::AgentBuilder,
    providers::OpenAIProvider,
};

let agent = AgentBuilder::new()
    .provider(OpenAIProvider::new(api_key))
    .prompt("You are a helpful assistant")
    .model("gpt-4o")
    .temperature(0.7)
    .build()?;

let response = agent.execute("Hello!").await?;
```

## Core Components

### 1. AgentBuilder

The builder provides a fluent API for constructing agents:

```rust
let agent = AgentBuilder::new()
    .provider(provider)              // Required: AI provider
    .prompt("System prompt")         // System instructions
    .model("model-name")            // Override default model
    .temperature(0.7)               // Control randomness
    .max_tokens(1000)               // Response length limit
    .max_iterations(5)              // Tool use iterations
    .memory(memory_store)           // Add memory capability
    .tool("name", executor)         // Add tools
    .context(existing_context)      // Start with context
    .build()?;
```

### 2. Context Management

Agents maintain conversation context automatically:

```rust
// Context is managed internally
agent.execute("What's 2+2?").await?;
agent.execute("Multiply that by 10").await?; // Remembers previous answer

// Clear context when needed
agent.clear_context();
```

Context features:
- Automatic message history
- System message preservation
- Token/message limits
- Metadata support

### 3. Tool System

Tools extend agent capabilities:

```rust
use lib_ai::agent::{ToolExecutor, ToolResult};

// Implement custom tools
struct WeatherTool;

#[async_trait]
impl ToolExecutor for WeatherTool {
    async fn execute(&self, args: &str) -> Result<ToolResult, Box<dyn Error>> {
        // Parse arguments and execute
        Ok(ToolResult::Success("Sunny, 72°F".to_string()))
    }
    
    fn definition(&self) -> ToolFunction {
        // Define tool schema
    }
}

// Add to agent
let agent = AgentBuilder::new()
    .tool("weather", WeatherTool)
    .build()?;
```

Built-in tools:
- `CalculatorTool` - Basic arithmetic
- `WebFetchTool` - Fetch URLs
- `KeyValueStoreTool` - Simple storage
- `FunctionTool` - Create tools from closures

### 4. Memory System

Agents can store and retrieve conversation history:

```rust
use lib_ai::agent::InMemoryStore;

let agent = AgentBuilder::new()
    .memory(InMemoryStore::new(100)) // Store last 100 exchanges
    .build()?;

// Memory is automatically used for context
agent.execute("Remember that my favorite color is blue").await?;
// Later...
agent.execute("What's my favorite color?").await?; // Will recall
```

Memory implementations:
- `InMemoryStore` - Simple in-memory storage
- `PersistentMemoryStore` - File-based persistence
- `SemanticMemoryStore` - Similarity-based retrieval

### 5. Streaming

Enable streaming for real-time responses:

```rust
let agent = AgentBuilder::new()
    .stream(true)
    .build()?;

let mut stream = agent.execute_stream("Tell me a story").await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(text) => print!("{}", text),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Advanced Usage

### Custom Tools with State

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

struct DatabaseTool {
    conn: Arc<Mutex<DatabaseConnection>>,
}

impl DatabaseTool {
    async fn query(&self, sql: &str) -> Result<Vec<Row>> {
        let conn = self.conn.lock().await;
        conn.execute(sql).await
    }
}
```

### Tool Registries

Manage multiple tools:

```rust
use lib_ai::agent::ToolRegistry;

let mut tools = ToolRegistry::new();
tools.register("calculator", CalculatorTool);
tools.register("weather", WeatherTool);
tools.register("database", DatabaseTool::new(conn));

let agent = AgentBuilder::new()
    .tools(tools)
    .build()?;
```

### Context Limits

Prevent context from growing too large:

```rust
use lib_ai::agent::Context;

let context = Context::with_limits(
    Some(50),    // Max 50 messages
    Some(4000),  // Max ~4000 tokens
);

let agent = AgentBuilder::new()
    .context(context)
    .build()?;
```

### Error Handling

```rust
use lib_ai::agent::AgentError;

match agent.execute("...").await {
    Ok(response) => println!("{}", response),
    Err(AgentError::ToolError(e)) => {
        eprintln!("Tool failed: {}", e);
    }
    Err(AgentError::ProviderError(e)) => {
        eprintln!("AI provider error: {}", e);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Examples

### Basic Chat Agent

```rust
let agent = AgentBuilder::new()
    .provider(provider)
    .prompt("You are a friendly chat assistant")
    .build()?;

loop {
    let input = read_user_input();
    let response = agent.chat(&input).await?;
    println!("AI: {}", response);
}
```

### Task-Oriented Agent

```rust
let agent = AgentBuilder::new()
    .provider(provider)
    .prompt("You are a task management assistant")
    .tool("create_task", CreateTaskTool)
    .tool("list_tasks", ListTasksTool)
    .tool("complete_task", CompleteTaskTool)
    .memory(PersistentMemoryStore::new("tasks.json", 1000)?)
    .build()?;

agent.execute("Create a task to buy groceries").await?;
```

### Research Agent

```rust
let agent = AgentBuilder::new()
    .provider(provider)
    .prompt("You are a research assistant. Gather and synthesize information.")
    .tool("web_search", WebSearchTool)
    .tool("fetch_page", WebFetchTool)
    .tool("summarize", SummarizeTool)
    .max_iterations(10) // Allow multiple tool uses
    .build()?;

let report = agent.execute("Research recent developments in quantum computing").await?;
```

### Multi-Provider Agent

```rust
// Use different providers for different tasks
let fast_agent = AgentBuilder::new()
    .provider(OpenAIProvider::new(key))
    .model("gpt-3.5-turbo")
    .build()?;

let powerful_agent = AgentBuilder::new()
    .provider(AnthropicProvider::new(key))
    .model("claude-3-opus-20240229")
    .build()?;
```

## Best Practices

1. **System Prompts**: Be specific about the agent's role and capabilities
2. **Tool Design**: Keep tools focused on single responsibilities
3. **Memory Management**: Set appropriate limits to prevent unbounded growth
4. **Error Recovery**: Handle tool failures gracefully
5. **Context Preservation**: Clear context between unrelated conversations
6. **Streaming**: Use for long responses or real-time interaction
7. **Tool Documentation**: Provide clear descriptions for tool discovery

## Performance Considerations

- **Token Usage**: Monitor context size to control costs
- **Memory Retrieval**: Use semantic search for large memory stores
- **Tool Execution**: Implement timeouts for long-running tools
- **Streaming**: Reduces time-to-first-token for better UX
- **Caching**: Consider caching tool results when appropriate

## Security

- **Tool Sandboxing**: Validate and sanitize tool inputs
- **API Key Management**: Use environment variables
- **Memory Privacy**: Encrypt sensitive stored conversations
- **Input Validation**: Sanitize user inputs before processing
- **Rate Limiting**: Implement per-user rate limits

## Observability

The agent system includes comprehensive observability features for monitoring, debugging, and cost tracking:

### Metrics Collection

```rust
use lib_ai::observability::MetricsCollector;

let metrics = Arc::new(MetricsCollector::new());

let agent = AgentBuilder::new()
    .provider(provider)
    .with_metrics(metrics.clone())
    .build()?;

// Execute operations
agent.execute("Analyze this data").await?;

// Access metrics
let agent_metrics = metrics.get_agent_metrics(agent.agent_id());
println!("Total requests: {}", agent_metrics.total_requests);
println!("Success rate: {:.2}%", 
    (agent_metrics.successful_requests as f64 / agent_metrics.total_requests as f64) * 100.0
);
println!("Average response time: {:?}", agent_metrics.average_response_time);

// Tool-specific metrics
for (tool_name, tool_metrics) in &agent_metrics.tool_usage {
    println!("{}: {} calls, {:?} avg duration", 
        tool_name, 
        tool_metrics.executions,
        tool_metrics.average_duration
    );
}
```

### Cost Tracking

Track API costs across providers:

```rust
use lib_ai::observability::CostTracker;

let cost_tracker = Arc::new(std::sync::RwLock::new(CostTracker::new()));

let agent = AgentBuilder::new()
    .provider(provider)
    .with_cost_tracker(cost_tracker.clone())
    .build()?;

// Later, generate cost reports
let report = cost_tracker.read().unwrap().generate_report();
println!("Total cost: ${:.6}", report.total_cost);

for provider in &report.providers {
    println!("{}: ${:.6}", provider.provider_name, provider.total_cost);
    for model in &provider.models {
        println!("  {}: ${:.6} ({} requests)", 
            model.model_name, 
            model.total_cost,
            model.request_count
        );
    }
}
```

### Tracing

Detailed execution tracing for debugging:

```rust
use lib_ai::observability::{AgentTracer, TracingConfig};

let tracer = Arc::new(AgentTracer::new(TracingConfig {
    enabled: true,
    sample_rate: 1.0, // Trace all requests
    max_traces: 1000,
    max_spans_per_trace: 100,
    export_interval: Duration::from_secs(30),
}));

let agent = AgentBuilder::new()
    .provider(provider)
    .with_tracer(tracer.clone())
    .build()?;

// Access traces
let traces = tracer.get_all_traces();
for (trace_id, events) in traces {
    println!("Trace {}: {} events", trace_id, events.len());
    for event in events {
        println!("  {}: {:?}", event.operation_name, event.duration);
    }
}
```

### Full Observability

Enable all observability features at once:

```rust
use lib_ai::observability::{
    MetricsCollector, AgentTracer, CostTracker, 
    TelemetryExporter, TelemetryConfig, ExporterConfig
};

// Set up observability components
let metrics = Arc::new(MetricsCollector::new());
let tracer = Arc::new(AgentTracer::new(tracing_config));
let cost_tracker = Arc::new(std::sync::RwLock::new(CostTracker::new()));

// Configure telemetry export
let telemetry_config = TelemetryConfig {
    enabled: true,
    export_interval: Duration::from_secs(60),
    exporters: vec![
        ExporterConfig {
            name: "console".to_string(),
            enabled: true,
            exporter_type: ExporterType::Console,
            endpoint: None,
            headers: HashMap::new(),
        },
        ExporterConfig {
            name: "prometheus".to_string(),
            enabled: true,
            exporter_type: ExporterType::Prometheus { 
                endpoint: "http://localhost:9090".to_string() 
            },
            endpoint: Some("http://localhost:9090".to_string()),
            headers: HashMap::new(),
        },
    ],
    batch_size: 100,
    max_queue_size: 1000,
};

let telemetry = Arc::new(TelemetryExporter::new(
    telemetry_config,
    metrics.clone(),
    tracer.clone(),
    Arc::new(tokio::sync::RwLock::new(cost_tracker.read().unwrap().clone())),
));

// Create agent with full observability
let agent = AgentBuilder::new()
    .provider(provider)
    .with_observability(
        metrics.clone(),
        tracer.clone(),
        cost_tracker.clone(),
        telemetry.clone(),
    )
    .build()?;
```

### Custom Metrics

Implement custom metrics collection:

```rust
use lib_ai::observability::MetricsCollector;

// Track custom metrics
metrics.record_custom_metric("cache_hit_rate", 0.85);
metrics.increment_counter("api_calls", 1);
metrics.record_histogram("response_size_bytes", 1024);
```

## Extending the System

The agent system is designed to be extensible:

1. **Custom Memory Stores**: Implement the `Memory` trait
2. **Custom Tools**: Implement the `ToolExecutor` trait
3. **Context Processors**: Pre/post-process messages
4. **Custom Builders**: Extend `AgentBuilder` for domain-specific agents
5. **Custom Metrics**: Add domain-specific observability
6. **Custom Exporters**: Implement the `Exporter` trait for telemetry

## Troubleshooting

Common issues and solutions:

1. **"Maximum iterations reached"**: Increase `max_iterations` or simplify task
2. **Memory not working**: Ensure memory store is mutable (`let mut agent`)
3. **Tools not being called**: Check tool definitions match provider format
4. **Context too large**: Implement context limits or summarization
5. **Streaming not working**: Ensure provider and model support streaming