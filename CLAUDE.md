# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Building and Testing
```bash
# Build the project
cargo build

# Run all tests (requires API keys)
cargo test

# Run tests without API calls
cargo test integration_tests mock_tests

# Run specific provider tests
OPENAI_API_KEY=your-key cargo test openai_tests

# Run tests with rate limiting
RUST_TEST_THREADS=1 cargo test

# Use the test script
./run_tests.sh

# Run a specific example
cargo run --example agent_basic
cargo run --example tool_calling

# Lint and check
cargo clippy
cargo fmt --check
```

### Development Workflow
```bash
# Run a single test
cargo test test_name -- --nocapture

# Check for compilation errors
cargo check

# Format code
cargo fmt

# Update dependencies
cargo update
```

## Architecture Overview

lib_ai is a Rust library providing a unified interface for multiple AI providers with an advanced agent system. The architecture follows these key patterns:

### Provider Abstraction
All AI providers implement the `CompletionProvider` trait, enabling seamless switching between providers. The trait-based design allows for dynamic dispatch and provider-agnostic code.

### Agent System
The agent system uses a builder pattern (`AgentBuilder`) to construct AI agents with:
- Tool execution framework for extending capabilities
- Memory systems (in-memory, semantic, persistent via SurrealDB)
- Context management for maintaining conversation state
- Streaming support for real-time responses

### Module Organization
- `src/providers/`: Individual provider implementations (OpenAI, Anthropic, Gemini, etc.)
- `src/agent/`: Agent system with tools and memory subsystems
- `src/embeddings/`: Embedding providers for semantic search
- `src/observability/`: Metrics, tracing, and cost tracking
- `src/traits.rs`: Core trait definitions
- `src/models.rs`: Shared data structures

### Async Architecture
The entire library is async-first using Tokio. All provider methods return futures, and streaming uses `tokio_stream`.

### Error Handling
Comprehensive error types using `thiserror` with provider-specific error variants. All operations return `Result<T, LibAIError>`.

## Key Implementation Details

### Adding a New Provider
1. Create a new file in `src/providers/`
2. Implement the `CompletionProvider` trait
3. Add provider-specific error variants to `LibAIError`
4. Export from `src/providers/mod.rs`
5. Add integration tests in `tests/`

### Tool Implementation
Tools implement the `ToolFunction` trait with:
- `name()`: Unique identifier
- `description()`: What the tool does
- `parameters()`: JSON schema for parameters
- `execute()`: Async function that performs the action

### Memory Systems
- `InMemoryStore`: Simple conversation history
- `SemanticMemory`: Embedding-based retrieval
- `SurrealDBMemory`: Persistent storage with SurrealDB

## Testing Strategy

Tests are organized by scope:
- Unit tests: No API calls, test internal logic
- Mock tests: HTTP mocking with `mockito`/`wiremock`
- Integration tests: Real API calls (require API keys)

Always run `cargo test integration_tests mock_tests` before committing to ensure no regressions without incurring API costs.