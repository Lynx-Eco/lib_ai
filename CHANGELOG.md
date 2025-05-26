# Changelog

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