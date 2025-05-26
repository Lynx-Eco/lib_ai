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

## Extending the System

The agent system is designed to be extensible:

1. **Custom Memory Stores**: Implement the `Memory` trait
2. **Custom Tools**: Implement the `ToolExecutor` trait
3. **Context Processors**: Pre/post-process messages
4. **Custom Builders**: Extend `AgentBuilder` for domain-specific agents

## Troubleshooting

Common issues and solutions:

1. **"Maximum iterations reached"**: Increase `max_iterations` or simplify task
2. **Memory not working**: Ensure memory store is mutable (`let mut agent`)
3. **Tools not being called**: Check tool definitions match provider format
4. **Context too large**: Implement context limits or summarization
5. **Streaming not working**: Ensure provider and model support streaming