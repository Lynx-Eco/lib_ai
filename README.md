# lib_ai

A generalized Rust client library for multiple AI providers with a powerful agent system.

## Supported Providers

- **Anthropic** (Claude models)
- **OpenAI** (GPT models)
- **Google Gemini**
- **xAI** (Grok models)
- **OpenRouter** (Multiple providers)
- **Cohere** (Command models)
- **Ollama** (Local models)
- **Replicate** (Cloud models)
- **Together AI** (Open source models)

## Basic Usage

```rust
use lib_ai::{providers::*, CompletionProvider, CompletionRequest, Message, Role, MessageContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a provider
    let provider = OpenAIProvider::new("your-api-key".to_string());
    
    // Build a request
    let request = CompletionRequest {
        model: provider.default_model().to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful assistant."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("Hello!"),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(150),
        stream: Some(false),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        json_schema: None,
    };
    
    // Get completion
    let response = provider.complete(request).await?;
    if let Some(text) = response.choices[0].message.content.as_text() {
        println!("Response: {}", text);
    }
    
    Ok(())
}
```

## Features

- **Unified interface** across all providers
- **Agent system** with tool use, memory, and context management
- **Streaming support** for real-time responses
- **Tool/Function calling** for extending AI capabilities
- **Structured output** with JSON mode and derive macros
- **Multimodal support** for text and images
- **Async/await** based
- **Type-safe** request/response models
- **Comprehensive error handling** with retry logic and circuit breakers
- **Observability** with metrics, tracing, and cost tracking
- **Memory systems** including semantic search and persistent storage

## Advanced Features

### Tool Calling

```rust
use lib_ai::{Tool, ToolType, ToolFunction, ToolChoice};
use serde_json::json;

// Define a tool
let weather_tool = Tool {
    r#type: ToolType::Function,
    function: ToolFunction {
        name: "get_weather".to_string(),
        description: Some("Get the current weather".to_string()),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            },
            "required": ["location"]
        }),
    },
};

// Use in request
let request = CompletionRequest {
    // ... other fields ...
    tools: Some(vec![weather_tool]),
    tool_choice: Some(ToolChoice::String("auto".to_string())),
    // ...
};
```

### Structured Output

```rust
use lib_ai::{ResponseFormat, ResponseFormatType};

// Request JSON output
let request = CompletionRequest {
    // ... other fields ...
    response_format: Some(ResponseFormat {
        r#type: ResponseFormatType::JsonObject,
    }),
    // ...
};
```

### Multimodal (Vision)

```rust
use lib_ai::{MessageContent, ContentPart, ImageUrl};

// Create a message with image
let message = Message {
    role: Role::User,
    content: MessageContent::Parts(vec![
        ContentPart::Text {
            text: "What's in this image?".to_string(),
        },
        ContentPart::Image {
            image_url: ImageUrl {
                url: "https://example.com/image.jpg".to_string(),
                detail: Some("high".to_string()),
            },
        },
    ]),
    tool_calls: None,
    tool_call_id: None,
};
```

### Agent System

Build intelligent agents with tools and memory:

```rust
use lib_ai::agent::{AgentBuilder, CalculatorTool, InMemoryStore};

let mut agent = AgentBuilder::new()
    .provider(provider)
    .prompt("You are a helpful assistant with a calculator")
    .tool("calculator", CalculatorTool)
    .memory(InMemoryStore::new(100))
    .build()?;

let response = agent.execute("What's 42 * 17?").await?;
```

See [AGENT.md](AGENT.md) for comprehensive agent documentation.

### Streaming

```rust
use futures::StreamExt;

let mut stream = provider.complete_stream(request).await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(chunk) => {
            for choice in chunk.choices {
                if let Some(content) = choice.delta.content {
                    print!("{}", content);
                }
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Environment Variables

Set the appropriate API key for your provider:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`
- `OPENROUTER_API_KEY`
- `COHERE_API_KEY`
- `REPLICATE_API_TOKEN`
- `TOGETHER_API_KEY`

For Ollama, no API key is needed as it runs locally.

## Testing

The library includes comprehensive test suites for each provider:

### Running Tests

```bash
# Run all tests (requires API keys)
cargo test

# Run only unit/integration tests (no API keys needed)
cargo test integration_tests
cargo test mock_tests

# Run tests for specific provider
OPENAI_API_KEY=your-key cargo test openai_tests

# Run with the test script
./run_tests.sh
```

### Test Coverage

- **Unit tests**: Core functionality without API calls
- **Mock tests**: API behavior with mocked responses
- **Integration tests**: Real API calls (requires API keys)
- **Provider tests**: Specific features for each provider

### Test Categories

1. **Basic completion**: Simple text generation
2. **Streaming**: Real-time response streaming
3. **Tool calling**: Function/tool invocation
4. **JSON mode**: Structured output
5. **Multimodal**: Image + text inputs
6. **Conversation**: Multi-turn dialogues
7. **Error handling**: Rate limits, invalid requests
8. **Model switching**: Different models per provider

## Observability

The library includes comprehensive observability features:

### Metrics Collection

```rust
use lib_ai::observability::MetricsCollector;

let metrics = Arc::new(MetricsCollector::new());
let agent = AgentBuilder::new()
    .provider(provider)
    .with_metrics(metrics.clone())
    .build()?;

// Access metrics
let agent_metrics = metrics.get_agent_metrics(agent.agent_id());
println!("Total requests: {}", agent_metrics.total_requests);
println!("Average response time: {:?}", agent_metrics.average_response_time);
```

### Cost Tracking

```rust
use lib_ai::observability::CostTracker;

let cost_tracker = Arc::new(std::sync::RwLock::new(CostTracker::new()));
let report = cost_tracker.read().unwrap().generate_report();
println!("Total cost: ${:.2}", report.total_cost);
```

### Tracing

```rust
use lib_ai::observability::{AgentTracer, TracingConfig};

let tracer = Arc::new(AgentTracer::new(TracingConfig {
    enabled: true,
    sample_rate: 1.0,
    max_traces: 1000,
    max_spans_per_trace: 100,
    export_interval: Duration::from_secs(30),
}));
```

## Error Handling

The library provides comprehensive error handling with retry logic and circuit breakers:

### Retry Logic

```rust
use lib_ai::error::{RetryConfig, RetryExecutor};

let retry_config = RetryConfig {
    max_attempts: 3,
    initial_delay: Duration::from_secs(1),
    max_delay: Duration::from_secs(60),
    backoff: BackoffStrategy::Exponential { multiplier: 2.0 },
    jitter: JitterStrategy::Full,
    respect_retry_after: true,
    max_total_time: Some(Duration::from_secs(300)),
    retry_condition: RetryCondition::Default,
};

let executor = RetryExecutor::new(retry_config);
let result = executor.execute(|| async {
    provider.complete(request).await
}).await?;
```

### Circuit Breaker

```rust
use lib_ai::error::{CircuitBreaker, CircuitBreakerConfig};

let circuit_breaker = CircuitBreaker::new("openai", CircuitBreakerConfig {
    failure_threshold: 50.0,
    minimum_request_count: 10,
    measurement_window: Duration::from_secs(60),
    recovery_timeout: Duration::from_secs(30),
    half_open_max_requests: 3,
    success_threshold: 60.0,
});

let result = circuit_breaker.execute(|| async {
    provider.complete(request).await
}).await?;
```

## Derive Macros

Enable the `derive` feature to use derive macros for structured output:

```toml
[dependencies]
lib_ai = { version = "0.1.0", features = ["derive"] }
```

```rust
use lib_ai_derive::StructuredOutput;

#[derive(Debug, Serialize, Deserialize, StructuredOutput)]
struct WeatherInfo {
    temperature: f32,
    humidity: u8,
    conditions: String,
}

let agent = AgentBuilder::new()
    .provider(provider)
    .structured::<WeatherInfo>()
    .build()?;

let weather: WeatherInfo = agent.complete_structured("What's the weather?").await?;
```