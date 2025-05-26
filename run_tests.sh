#!/bin/bash

# Script to run tests for lib_ai with optional API keys

echo "🧪 Running lib_ai test suite..."
echo ""

# Check which API keys are available
available_providers=()

if [ ! -z "$OPENAI_API_KEY" ]; then
    available_providers+=("OpenAI")
fi

if [ ! -z "$ANTHROPIC_API_KEY" ]; then
    available_providers+=("Anthropic")
fi

if [ ! -z "$GEMINI_API_KEY" ]; then
    available_providers+=("Gemini")
fi

if [ ! -z "$XAI_API_KEY" ]; then
    available_providers+=("xAI")
fi

if [ ! -z "$OPENROUTER_API_KEY" ]; then
    available_providers+=("OpenRouter")
fi

echo "Available providers: ${available_providers[@]}"
echo ""

# Run integration tests (no API keys needed)
echo "Running integration tests..."
cargo test integration_tests -- --nocapture

# Run provider-specific tests
if [ ${#available_providers[@]} -eq 0 ]; then
    echo ""
    echo "⚠️  No API keys found. Skipping provider-specific tests."
    echo ""
    echo "To run provider tests, set one or more of these environment variables:"
    echo "  - OPENAI_API_KEY"
    echo "  - ANTHROPIC_API_KEY"
    echo "  - GEMINI_API_KEY"
    echo "  - XAI_API_KEY"
    echo "  - OPENROUTER_API_KEY"
else
    echo ""
    echo "Running provider-specific tests..."
    
    # Run tests with rate limiting to avoid hitting API limits
    RUST_TEST_THREADS=1 cargo test --test '*_tests' -- --nocapture
fi

echo ""
echo "✅ Test suite complete!"