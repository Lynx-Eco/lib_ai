# Error Handling Documentation

The lib_ai library provides a comprehensive error handling system with retry logic, circuit breakers, and detailed error types for robust AI operations.

## Error Types

The library defines a rich set of error types in the `AiError` enum:

### Network and Connection Errors

```rust
// Network request failed
AiError::NetworkError { 
    message: String, 
    retryable: bool,
    status_code: Option<u16>,
}

// Connection timeout
AiError::TimeoutError { 
    timeout: Duration,
    retryable: bool,
}

// Connection refused
AiError::ConnectionRefused { endpoint: String }
```

### Authentication Errors

```rust
// Invalid API key
AiError::InvalidApiKey { provider: String }

// Authentication failed
AiError::AuthenticationFailed { reason: String }

// API key expired
AiError::ApiKeyExpired { provider: String }
```

### Rate Limiting Errors

```rust
// Rate limit exceeded
AiError::RateLimitExceeded { 
    retry_after: Option<Duration>,
    daily_limit: Option<u64>,
    requests_remaining: Option<u64>,
}

// Quota exceeded
AiError::QuotaExceeded { 
    provider: String,
    quota_type: String, // "monthly", "daily", "requests", "tokens"
    reset_time: Option<SystemTime>,
}
```

### Request/Response Errors

```rust
// Invalid request
AiError::InvalidRequest { 
    message: String,
    field: Option<String>,
    code: Option<String>,
}

// Malformed response
AiError::MalformedResponse { 
    message: String,
    raw_response: Option<String>,
}

// Unsupported model
AiError::UnsupportedModel { 
    model: String,
    provider: String,
    available_models: Vec<String>,
}
```

### Provider-Specific Errors

```rust
// Provider error
AiError::ProviderError { 
    provider: String,
    message: String,
    error_code: Option<String>,
    retryable: bool,
}

// Service unavailable
AiError::ServiceUnavailable { 
    provider: String,
    retry_after: Option<Duration>,
}
```

## Error Properties

Each error type has useful properties:

```rust
// Check if error is retryable
if error.is_retryable() {
    // Retry the operation
}

// Get retry delay if specified
if let Some(delay) = error.retry_after() {
    sleep(delay).await;
}

// Get error severity
match error.severity() {
    ErrorSeverity::Critical => // Alert immediately
    ErrorSeverity::High => // Log and monitor
    ErrorSeverity::Medium => // Standard handling
    ErrorSeverity::Low => // Informational
}

// Get associated provider
if let Some(provider) = error.provider() {
    println!("Error from provider: {}", provider);
}
```

## Retry Logic

The library includes a powerful retry system with configurable strategies:

### Basic Retry

```rust
use lib_ai::error::{retry_with_default, RetryExecutor, RetryConfig};

// Simple retry with defaults
let result = retry_with_default(|| async {
    provider.complete(request).await
}).await?;
```

### Custom Retry Configuration

```rust
use lib_ai::error::{RetryConfig, BackoffStrategy, JitterStrategy};

let retry_config = RetryConfig {
    max_attempts: 5,
    initial_delay: Duration::from_millis(500),
    max_delay: Duration::from_secs(30),
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

### Backoff Strategies

```rust
// Fixed delay between retries
BackoffStrategy::Fixed

// Linear backoff: delay = initial_delay * attempt
BackoffStrategy::Linear

// Exponential backoff: delay = initial_delay * multiplier^attempt
BackoffStrategy::Exponential { multiplier: 2.0 }

// Custom delays for each attempt
BackoffStrategy::Custom(vec![
    Duration::from_secs(1),
    Duration::from_secs(3),
    Duration::from_secs(10),
])
```

### Jitter Strategies

Jitter helps avoid the "thundering herd" problem:

```rust
// No jitter
JitterStrategy::None

// Random jitter up to the full delay
JitterStrategy::Full

// Jitter up to half the delay
JitterStrategy::Half

// Fixed amount of jitter
JitterStrategy::Fixed(Duration::from_millis(100))

// Decorrelated jitter (AWS recommended)
JitterStrategy::Decorrelated
```

### Retry Conditions

Control when retries occur:

```rust
// Use default retry logic based on error type
RetryCondition::Default

// Always retry (up to max attempts)
RetryCondition::Always

// Never retry
RetryCondition::Never

// Retry only specific error types
RetryCondition::ErrorTypes(vec![
    "NetworkError".to_string(),
    "TimeoutError".to_string(),
])
```

### Retry Context

Access retry context during execution:

```rust
let executor = RetryExecutor::new(config);
let result = executor.execute(|| async {
    // Access context if needed
    println!("Attempt: {}", context.attempt);
    println!("Total elapsed: {:?}", context.total_elapsed);
    
    provider.complete(request).await
}).await?;
```

## Circuit Breaker

Prevent cascading failures with circuit breakers:

### Basic Circuit Breaker

```rust
use lib_ai::error::{CircuitBreaker, CircuitBreakerConfig};

let circuit_breaker = CircuitBreaker::new("openai", CircuitBreakerConfig::default());

let result = circuit_breaker.execute(|| async {
    provider.complete(request).await
}).await?;
```

### Custom Configuration

```rust
let config = CircuitBreakerConfig {
    failure_threshold: 50.0,          // Open circuit at 50% failure rate
    minimum_request_count: 10,        // Need at least 10 requests
    measurement_window: Duration::from_secs(60),
    recovery_timeout: Duration::from_secs(30),
    half_open_max_requests: 3,
    success_threshold: 60.0,          // 60% success to close circuit
};

let circuit_breaker = CircuitBreaker::new("anthropic", config);
```

### Circuit States

The circuit breaker has three states:

1. **Closed**: Normal operation, requests pass through
2. **Open**: Failures exceeded threshold, requests are rejected
3. **Half-Open**: Testing if service recovered, limited requests allowed

```rust
// Check circuit state
match circuit_breaker.state() {
    CircuitState::Closed => println!("Circuit is healthy"),
    CircuitState::Open { opened_at } => {
        println!("Circuit opened at: {:?}", opened_at);
    }
    CircuitState::HalfOpen { attempts, successes, .. } => {
        println!("Testing recovery: {}/{} successful", successes, attempts);
    }
}

// Get metrics
let metrics = circuit_breaker.metrics();
println!("Failure rate: {:.2}%", metrics.failure_rate);
println!("Total requests: {}", metrics.total_requests);
```

### Manual Control

```rust
// Manually open the circuit (e.g., for maintenance)
circuit_breaker.open();

// Reset to closed state
circuit_breaker.reset();

// Force close
circuit_breaker.close();
```

## Circuit Breaker Registry

Manage multiple circuit breakers:

```rust
use lib_ai::error::CircuitBreakerRegistry;

let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

// Get or create circuit breakers
let openai_cb = registry.get_or_create("openai");
let anthropic_cb = registry.get_or_create("anthropic");

// Create with custom config
let gemini_cb = registry.create_with_config("gemini", custom_config);

// Get all metrics
let all_metrics = registry.get_all_metrics();
for metric in all_metrics {
    println!("{}: {:.2}% failure rate", 
        metric.service_name, 
        metric.failure_rate
    );
}

// Reset all circuits
registry.reset_all();
```

## Resilient Provider

Combine retry logic and circuit breakers:

```rust
use lib_ai::error::{ResilientProvider, ResilientProviderBuilder};

// Create with defaults
let resilient = ResilientProvider::new(Arc::new(provider));

// Custom configuration
let resilient = ResilientProviderBuilder::new()
    .max_retries(5)
    .failure_threshold(75.0)
    .recovery_timeout(Duration::from_secs(60))
    .build(Arc::new(provider));

// Use like any provider
let response = resilient.complete(request).await?;
```

## Error Handling Patterns

### Pattern 1: Graceful Degradation

```rust
let primary = ResilientProvider::new(Arc::new(OpenAIProvider::new(key)));
let fallback = ResilientProvider::new(Arc::new(AnthropicProvider::new(key)));

let response = match primary.complete(request.clone()).await {
    Ok(response) => response,
    Err(e) if e.is_retryable() => {
        warn!("Primary provider failed, using fallback: {}", e);
        fallback.complete(request).await?
    }
    Err(e) => return Err(e),
};
```

### Pattern 2: Retry with Different Models

```rust
let models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"];

for model in models {
    let mut request = request.clone();
    request.model = model.to_string();
    
    match provider.complete(request).await {
        Ok(response) => return Ok(response),
        Err(AiError::UnsupportedModel { .. }) => continue,
        Err(e) => return Err(e),
    }
}
```

### Pattern 3: Cost-Aware Retry

```rust
let expensive_limit = 3;
let cheap_limit = 10;

let retry_config = if model.contains("gpt-4") {
    RetryConfig {
        max_attempts: expensive_limit,
        ..Default::default()
    }
} else {
    RetryConfig {
        max_attempts: cheap_limit,
        ..Default::default()
    }
};
```

### Pattern 4: Observability Integration

```rust
match provider.complete(request).await {
    Ok(response) => {
        metrics.record_success();
        Ok(response)
    }
    Err(e) => {
        metrics.record_failure(&e);
        
        if let Some(delay) = e.retry_after() {
            metrics.record_retry_delay(delay);
        }
        
        Err(e)
    }
}
```

## Best Practices

1. **Always check `is_retryable()`** before retrying operations
2. **Respect `retry_after()`** delays from rate limit errors
3. **Use circuit breakers** for external services to prevent cascading failures
4. **Configure appropriate timeouts** to fail fast
5. **Log errors with context** for debugging
6. **Monitor error rates** and adjust thresholds accordingly
7. **Test error handling** with mock failures
8. **Document expected errors** in your API

## Testing Error Handling

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_retry_logic() {
        let mut attempt = 0;
        
        let result = retry_with_default(|| async {
            attempt += 1;
            if attempt < 3 {
                Err(AiError::NetworkError {
                    message: "Temporary failure".to_string(),
                    retryable: true,
                    status_code: Some(503),
                })
            } else {
                Ok("Success".to_string())
            }
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(attempt, 3);
    }
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let cb = CircuitBreaker::new("test", CircuitBreakerConfig {
            failure_threshold: 50.0,
            minimum_request_count: 2,
            ..Default::default()
        });
        
        // First failure
        let _ = cb.execute(|| async {
            Err::<(), _>(AiError::NetworkError {
                message: "Failed".to_string(),
                retryable: true,
                status_code: Some(500),
            })
        }).await;
        
        // Second failure opens circuit
        let _ = cb.execute(|| async {
            Err::<(), _>(AiError::NetworkError {
                message: "Failed".to_string(),
                retryable: true,
                status_code: Some(500),
            })
        }).await;
        
        // Circuit should be open
        assert!(matches!(cb.state(), CircuitState::Open { .. }));
    }
}
```

## Troubleshooting

Common issues and solutions:

1. **"Circuit breaker keeps opening"**
   - Check failure threshold and measurement window
   - Verify the service is actually healthy
   - Consider increasing `minimum_request_count`

2. **"Retries happening too fast"**
   - Add jitter to prevent thundering herd
   - Increase `initial_delay`
   - Use exponential backoff

3. **"Not retrying when expected"**
   - Check `is_retryable()` for the error type
   - Verify `retry_condition` configuration
   - Check `max_total_time` hasn't been exceeded

4. **"Circuit never recovers"**
   - Adjust `recovery_timeout`
   - Increase `half_open_max_requests`
   - Lower `success_threshold`