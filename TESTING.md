# lib_ai Test Suite Documentation

## Overview

The lib_ai library includes a comprehensive test suite covering unit tests, integration tests, and provider-specific tests.

## Test Structure

```
tests/
├── common/
│   └── mod.rs           # Shared test utilities and request builders
├── integration_tests.rs # Tests for cross-provider functionality
├── mock_tests.rs        # Tests using mocked HTTP responses
├── openai_tests.rs      # OpenAI-specific integration tests
├── anthropic_tests.rs   # Anthropic-specific integration tests
├── gemini_tests.rs      # Gemini-specific integration tests
├── xai_tests.rs         # xAI-specific integration tests
└── openrouter_tests.rs  # OpenRouter-specific integration tests
```

## Running Tests

### Quick Start

```bash
# Run all tests (requires API keys)
cargo test

# Run only tests that don't require API keys
cargo test integration_tests mock_tests

# Run with the provided script
./run_tests.sh
```

### Provider-Specific Tests

To run tests for a specific provider, set the corresponding API key:

```bash
# OpenAI
OPENAI_API_KEY=your-key cargo test openai_tests

# Anthropic
ANTHROPIC_API_KEY=your-key cargo test anthropic_tests

# Gemini
GEMINI_API_KEY=your-key cargo test gemini_tests

# xAI
XAI_API_KEY=your-key cargo test xai_tests

# OpenRouter
OPENROUTER_API_KEY=your-key cargo test openrouter_tests
```

### Running Tests with Rate Limiting

To avoid hitting API rate limits when running all tests:

```bash
RUST_TEST_THREADS=1 cargo test
```

## Test Categories

### 1. Unit/Integration Tests (`integration_tests.rs`)

Tests that verify the library's internal consistency without making API calls:

- **test_provider_trait_consistency**: Ensures all providers implement the trait correctly
- **test_dynamic_provider_switching**: Tests dynamic dispatch with trait objects
- **test_message_content_variants**: Tests MessageContent enum serialization
- **test_tool_call_creation**: Tests tool call structure creation
- **test_error_handling**: Tests error type consistency
- **test_request_builder_pattern**: Tests request construction
- **test_model_info**: Verifies model availability consistency

### 2. Mock Tests (`mock_tests.rs`)

Tests using mocked HTTP responses to verify request/response handling:

- **test_openai_error_response**: Tests error response parsing
- **test_openai_success_response**: Tests successful completion parsing
- **test_streaming_response**: Tests SSE stream parsing
- **test_tool_calling_response**: Tests tool call response parsing
- **test_connection_error**: Tests network error handling

### 3. Provider Tests

Each provider has tests for:

- **Simple completion**: Basic text generation
- **Streaming**: Real-time response streaming
- **Tool calling**: Function/tool invocation (where supported)
- **JSON mode**: Structured output (where supported)
- **Multimodal**: Image + text inputs (where supported)
- **Conversation**: Multi-turn dialogues
- **Temperature control**: Deterministic vs random responses
- **Model switching**: Testing different models
- **Error handling**: Invalid models, rate limits

## Common Test Utilities

The `tests/common/mod.rs` module provides request builders:

- `create_simple_request()`: Basic completion request
- `create_streaming_request()`: Streaming completion request
- `create_tool_request()`: Request with tool definitions
- `create_json_request()`: Request for JSON output
- `create_multimodal_request()`: Request with image input
- `create_conversation_request()`: Multi-turn conversation

## Testing Best Practices

1. **API Keys**: Store API keys in environment variables, never commit them
2. **Rate Limits**: Use `RUST_TEST_THREADS=1` to avoid rate limits
3. **Costs**: Be aware that running tests incurs API costs
4. **Determinism**: Use `temperature: 0.0` for reproducible tests
5. **Timeouts**: Some providers may have longer response times

## Continuous Integration

For CI/CD pipelines:

```yaml
# Example GitHub Actions configuration
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  # ... other keys

steps:
  - name: Run tests
    run: |
      cargo test integration_tests mock_tests  # Always run
      if [ ! -z "$OPENAI_API_KEY" ]; then
        RUST_TEST_THREADS=1 cargo test openai_tests
      fi
      # ... similar for other providers
```

## Troubleshooting

### Common Issues

1. **"API key not set"**: Export the appropriate environment variable
2. **Rate limit errors**: Use `RUST_TEST_THREADS=1` or add delays
3. **Timeout errors**: Some providers may be slower, increase timeouts
4. **Model not available**: Check if the model is still supported

### Debug Output

Run tests with output to see responses:

```bash
cargo test test_name -- --nocapture
```

## Adding New Tests

When adding tests:

1. Use the common utilities for consistency
2. Test both success and error cases
3. Use deterministic settings (`temperature: 0.0`)
4. Add appropriate assertions
5. Document any provider-specific behavior

Example:

```rust
#[tokio::test]
async fn test_new_feature() {
    let Some(provider) = get_provider() else { return };
    
    let mut request = common::create_simple_request(provider.default_model().to_string());
    // Modify request for your test
    
    let response = provider.complete(request).await.unwrap();
    
    // Add assertions
    assert!(!response.id.is_empty());
    // ...
}
```