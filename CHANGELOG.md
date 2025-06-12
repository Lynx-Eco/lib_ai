# Changelog

## [Unreleased]

### Added

#### New Providers
- **Cohere**: Support for Command models with native API
- **Ollama**: Local model support for offline usage
- **Replicate**: Cloud model deployment platform
- **Together AI**: Open source model hosting

#### Observability Features
- **MetricsCollector**: Comprehensive metrics tracking
  - Request counts, success rates, response times
  - Token usage and cost tracking per provider/model
  - Tool execution metrics
  - Global and per-agent metrics
- **CostTracker**: Detailed cost analysis
  - Per-provider and per-model cost breakdown
  - Token usage tracking with pricing calculations
  - Cost report generation
- **AgentTracer**: Distributed tracing support
  - Trace events with timing information
  - Parent-child span relationships
  - Configurable sampling and retention
- **TelemetryExporter**: Unified telemetry export
  - Multiple export formats (Console, File, HTTP, Jaeger, Prometheus, OpenTelemetry)
  - Configurable batch export
  - Background export with intervals

#### Error Handling Enhancements
- **Comprehensive Error Types**: Rich error taxonomy
  - Network, timeout, and connection errors
  - Authentication and authorization errors
  - Rate limiting and quota errors
  - Provider-specific errors with metadata
- **Retry System**: Configurable retry logic
  - Multiple backoff strategies (Fixed, Linear, Exponential, Custom)
  - Jitter strategies to prevent thundering herd
  - Respect for retry-after headers
  - Maximum time limits
- **Circuit Breaker**: Fault tolerance
  - Configurable failure thresholds
  - Half-open state for recovery testing
  - Per-service circuit breakers
  - Registry for managing multiple breakers
- **ResilientProvider**: Combined retry and circuit breaker
  - Transparent resilience wrapper
  - Builder pattern for configuration

#### Derive Macros
- **StructuredOutput**: Derive macro for structured agent responses
- Automatic JSON schema generation
- Type-safe structured completions

#### Agent System Enhancements
- **Full observability integration**: Metrics, tracing, and cost tracking
- **with_observability()**: Single method to enable all observability
- **Structured agent support**: Type-safe responses with derive macros
- **Enhanced error handling**: Automatic retry and circuit breaking

### Changed
- **Error types**: Migrated from simple enums to structured variants with detailed fields
- **Provider error handling**: Improved error messages and retry logic
- **Test structure**: Separated mock tests from integration tests

### Fixed
- **Ollama provider**: Fixed lifetime issues with default_model
- **Together provider**: Fixed request/response type mismatches
- **Compilation errors**: Fixed error variant constructors in tests
- **Observability demo**: Fixed type mismatches and import errors

## [0.1.0] - Initial Release

### Features

#### Core Library
- Unified interface for multiple AI providers (OpenAI, Anthropic, Gemini, xAI, OpenRouter)
- Streaming support for real-time responses
- Tool/Function calling capabilities
- Structured output with JSON mode
- Multimodal support (text + images)
- Comprehensive error handling
- Full async/await support

#### Providers
- **OpenAI**: Full support including GPT-4, GPT-3.5, and o1 models
- **Anthropic**: Claude models with proper message formatting
- **Gemini**: Google's models with multimodal capabilities
- **xAI**: Grok models via OpenAI-compatible API
- **OpenRouter**: Multi-provider gateway with model listing

#### Agent System
- Fluent builder pattern for agent construction
- Tool execution framework with built-in and custom tools
- Memory systems (in-memory, persistent, semantic)
- Context management with limits and preservation
- Streaming support for agents
- Multi-turn conversations with state management

#### Built-in Tools
- `CalculatorTool`: Basic arithmetic operations
- `WebFetchTool`: HTTP request capabilities
- `KeyValueStoreTool`: Simple storage solution
- `FunctionTool`: Create tools from closures

### Documentation
- Comprehensive README with examples
- Agent system documentation (AGENT.md)
- Testing guide (TESTING.md)
- API examples for all features

### Testing
- Unit tests for core functionality
- Integration tests for cross-provider features
- Mock tests for HTTP interactions
- Provider-specific test suites
- Agent system tests

### Examples
- Basic completion usage
- Tool calling demonstrations
- Structured output examples
- Multimodal requests
- Agent creation and usage
- Custom tool implementation
- Interactive chat interface

### Fixed Issues
- Proper error handling for all providers
- Correct message role mapping
- Stream parsing for all providers
- Tool call format compatibility
- Memory persistence edge cases
- Context limit enforcement