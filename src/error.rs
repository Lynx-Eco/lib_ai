use async_trait::async_trait;
use futures::stream::Stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use tokio::time::sleep;

/// Comprehensive error types for AI operations
#[derive(Error, Debug, Clone)]
pub enum AiError {
    // Network and Connection Errors
    #[error("Network request failed: {message}")]
    NetworkError {
        message: String,
        retryable: bool,
        status_code: Option<u16>,
    },

    #[error("Connection timeout after {timeout:?}")]
    TimeoutError { timeout: Duration, retryable: bool },

    #[error("Connection refused to {endpoint}")]
    ConnectionRefused { endpoint: String },

    // Authentication Errors
    #[error("Invalid API key for provider {provider}")]
    InvalidApiKey { provider: String },

    #[error("Authentication failed: {reason}")]
    AuthenticationFailed { reason: String },

    #[error("API key expired for provider {provider}")]
    ApiKeyExpired { provider: String },

    // Rate Limiting Errors
    #[error("Rate limit exceeded. Retry after {retry_after:?}")]
    RateLimitExceeded {
        retry_after: Option<Duration>,
        daily_limit: Option<u64>,
        requests_remaining: Option<u64>,
    },

    #[error("Quota exceeded for provider {provider}")]
    QuotaExceeded {
        provider: String,
        quota_type: String, // "monthly", "daily", "requests", "tokens"
        reset_time: Option<SystemTime>,
    },

    // Request/Response Errors
    #[error("Invalid request: {message}")]
    InvalidRequest {
        message: String,
        field: Option<String>,
        code: Option<String>,
    },

    #[error("Malformed response: {message}")]
    MalformedResponse {
        message: String,
        raw_response: Option<String>,
    },

    #[error("Unsupported model {model} for provider {provider}")]
    UnsupportedModel {
        model: String,
        provider: String,
        available_models: Vec<String>,
    },

    // Provider-Specific Errors
    #[error("Provider error from {provider}: {message}")]
    ProviderError {
        provider: String,
        message: String,
        error_code: Option<String>,
        retryable: bool,
    },

    #[error("Service unavailable for provider {provider}")]
    ServiceUnavailable {
        provider: String,
        retry_after: Option<Duration>,
    },

    // Content and Safety Errors
    #[error("Content filtered by safety policies")]
    ContentFiltered {
        reason: String,
        category: Option<String>,
    },

    #[error("Request too large: {size} bytes (max: {max_size} bytes)")]
    RequestTooLarge { size: usize, max_size: usize },

    #[error("Response too large: {size} bytes")]
    ResponseTooLarge { size: usize },

    // Token and Usage Errors
    #[error("Token limit exceeded: {tokens} tokens (max: {max_tokens})")]
    TokenLimitExceeded { tokens: u32, max_tokens: u32 },

    #[error("Insufficient tokens: {available} available, {required} required")]
    InsufficientTokens { available: u32, required: u32 },

    // Serialization/Deserialization Errors
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("JSON parsing error: {message}")]
    JsonError {
        message: String,
        line: Option<usize>,
        column: Option<usize>,
    },

    // Stream and Async Errors
    #[error("Stream error: {message}")]
    StreamError { message: String, retryable: bool },

    #[error("Stream interrupted after {chunks_received} chunks")]
    StreamInterrupted { chunks_received: usize },

    // Tool and Function Errors
    #[error("Tool execution failed: {tool_name}")]
    ToolExecutionError {
        tool_name: String,
        message: String,
        retryable: bool,
    },

    #[error("Tool not found: {tool_name}")]
    ToolNotFound {
        tool_name: String,
        available_tools: Vec<String>,
    },

    #[error("Invalid tool parameters for {tool_name}: {message}")]
    InvalidToolParameters {
        tool_name: String,
        message: String,
        expected_schema: Option<String>,
    },

    // Memory and Context Errors
    #[error("Memory operation failed: {operation}")]
    MemoryError { operation: String, message: String },

    #[error("Context too large: {size} tokens (max: {max_size})")]
    ContextTooLarge { size: usize, max_size: usize },

    // Configuration Errors
    #[error("Invalid configuration: {field}")]
    ConfigurationError {
        field: String,
        message: String,
        suggestion: Option<String>,
    },

    #[error("Missing required configuration: {field}")]
    MissingConfiguration { field: String, description: String },

    // Circuit Breaker Errors
    #[error("Circuit breaker open for {service}. Failure rate: {failure_rate:.2}%")]
    CircuitBreakerOpen {
        service: String,
        failure_rate: f64,
        retry_after: Duration,
    },

    // Internal Errors
    #[error("Internal error: {message}")]
    InternalError {
        message: String,
        component: Option<String>,
    },

    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },

    // Custom errors for extensibility
    #[error("Custom error: {message}")]
    Custom {
        message: String,
        error_type: String,
        metadata: HashMap<String, String>,
    },
}

impl AiError {
    /// Check if this error type is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            AiError::NetworkError { retryable, .. } => *retryable,
            AiError::TimeoutError { retryable, .. } => *retryable,
            AiError::RateLimitExceeded { .. } => true,
            AiError::ServiceUnavailable { .. } => true,
            AiError::ProviderError { retryable, .. } => *retryable,
            AiError::StreamError { retryable, .. } => *retryable,
            AiError::ToolExecutionError { retryable, .. } => *retryable,
            AiError::StreamInterrupted { .. } => true,

            // Non-retryable errors
            AiError::InvalidApiKey { .. }
            | AiError::AuthenticationFailed { .. }
            | AiError::ApiKeyExpired { .. }
            | AiError::QuotaExceeded { .. }
            | AiError::InvalidRequest { .. }
            | AiError::UnsupportedModel { .. }
            | AiError::ContentFiltered { .. }
            | AiError::RequestTooLarge { .. }
            | AiError::TokenLimitExceeded { .. }
            | AiError::InsufficientTokens { .. }
            | AiError::SerializationError { .. }
            | AiError::JsonError { .. }
            | AiError::ToolNotFound { .. }
            | AiError::InvalidToolParameters { .. }
            | AiError::ConfigurationError { .. }
            | AiError::MissingConfiguration { .. }
            | AiError::NotImplemented { .. } => false,

            // Depends on specific context
            AiError::ConnectionRefused { .. } => true,
            AiError::MalformedResponse { .. } => false,
            AiError::ResponseTooLarge { .. } => false,
            AiError::MemoryError { .. } => false,
            AiError::ContextTooLarge { .. } => false,
            AiError::CircuitBreakerOpen { .. } => false, // Handle differently
            AiError::InternalError { .. } => true,
            AiError::Custom { metadata, .. } => {
                metadata.get("retryable").is_some_and(|v| v == "true")
            }
        }
    }

    /// Get retry delay if applicable
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            AiError::RateLimitExceeded { retry_after, .. } => *retry_after,
            AiError::ServiceUnavailable { retry_after, .. } => *retry_after,
            AiError::CircuitBreakerOpen { retry_after, .. } => Some(*retry_after),
            _ => None,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            AiError::InternalError { .. } | AiError::MemoryError { .. } => ErrorSeverity::Critical,

            AiError::InvalidApiKey { .. }
            | AiError::AuthenticationFailed { .. }
            | AiError::ApiKeyExpired { .. }
            | AiError::QuotaExceeded { .. }
            | AiError::ConfigurationError { .. }
            | AiError::MissingConfiguration { .. } => ErrorSeverity::High,

            AiError::NetworkError { .. }
            | AiError::TimeoutError { .. }
            | AiError::ConnectionRefused { .. }
            | AiError::RateLimitExceeded { .. }
            | AiError::ServiceUnavailable { .. }
            | AiError::CircuitBreakerOpen { .. }
            | AiError::StreamError { .. } => ErrorSeverity::Medium,

            AiError::InvalidRequest { .. }
            | AiError::UnsupportedModel { .. }
            | AiError::ContentFiltered { .. }
            | AiError::TokenLimitExceeded { .. }
            | AiError::ToolNotFound { .. }
            | AiError::InvalidToolParameters { .. } => ErrorSeverity::Low,

            _ => ErrorSeverity::Low,
        }
    }

    /// Get the provider associated with this error, if any
    pub fn provider(&self) -> Option<&str> {
        match self {
            AiError::InvalidApiKey { provider }
            | AiError::ApiKeyExpired { provider }
            | AiError::QuotaExceeded { provider, .. }
            | AiError::UnsupportedModel { provider, .. }
            | AiError::ProviderError { provider, .. }
            | AiError::ServiceUnavailable { provider, .. } => Some(provider),
            _ => None,
        }
    }

    /// Create a custom error with metadata
    pub fn custom(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        AiError::Custom {
            message: message.into(),
            error_type: error_type.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to a custom error
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let AiError::Custom {
            ref mut metadata, ..
        } = self
        {
            metadata.insert(key.into(), value.into());
        }
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl ErrorSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorSeverity::Low => "low",
            ErrorSeverity::Medium => "medium",
            ErrorSeverity::High => "high",
            ErrorSeverity::Critical => "critical",
        }
    }
}

// Conversion from common error types
impl From<reqwest::Error> for AiError {
    fn from(err: reqwest::Error) -> Self {
        let retryable = err.is_timeout()
            || err.is_connect()
            || err.status().is_some_and(|s| s.is_server_error());

        let status_code = err.status().map(|s| s.as_u16());

        if err.is_timeout() {
            AiError::TimeoutError {
                timeout: Duration::from_secs(30), // Default timeout
                retryable: true,
            }
        } else if err.is_connect() {
            AiError::ConnectionRefused {
                endpoint: err.url().map_or("unknown".to_string(), |u| u.to_string()),
            }
        } else {
            AiError::NetworkError {
                message: err.to_string(),
                retryable,
                status_code,
            }
        }
    }
}

impl From<serde_json::Error> for AiError {
    fn from(err: serde_json::Error) -> Self {
        AiError::JsonError {
            message: err.to_string(),
            line: Some(err.line()),
            column: Some(err.column()),
        }
    }
}

impl From<std::io::Error> for AiError {
    fn from(err: std::io::Error) -> Self {
        AiError::InternalError {
            message: err.to_string(),
            component: Some("io".to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, AiError>;

// RETRY LOGIC

/// Retry strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,

    /// Initial delay between retries
    pub initial_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Backoff strategy
    pub backoff: BackoffStrategy,

    /// Jitter strategy to avoid thundering herd
    pub jitter: JitterStrategy,

    /// Whether to respect retry-after headers
    pub respect_retry_after: bool,

    /// Maximum total time to spend retrying
    pub max_total_time: Option<Duration>,

    /// Custom retry condition
    pub retry_condition: RetryCondition,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(60),
            backoff: BackoffStrategy::Exponential { multiplier: 2.0 },
            jitter: JitterStrategy::Full,
            respect_retry_after: true,
            max_total_time: Some(Duration::from_secs(300)), // 5 minutes
            retry_condition: RetryCondition::Default,
        }
    }
}

/// Backoff strategies for retry delays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,

    /// Linear backoff: delay = initial_delay * attempt
    Linear,

    /// Exponential backoff: delay = initial_delay * multiplier^attempt
    Exponential { multiplier: f64 },

    /// Custom delays for each attempt
    Custom(Vec<Duration>),
}

/// Jitter strategies to randomize retry delays
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JitterStrategy {
    /// No jitter
    None,

    /// Add random jitter up to the full delay
    Full,

    /// Add jitter up to half the delay
    Half,

    /// Add fixed amount of jitter
    Fixed(Duration),

    /// Custom jitter function
    Decorrelated,
}

/// Conditions for when to retry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryCondition {
    /// Use default retry logic based on error type
    Default,

    /// Always retry
    Always,

    /// Never retry
    Never,

    /// Retry only specific error types
    ErrorTypes(Vec<String>),

    /// Custom condition function
    Custom,
}

/// Retry execution context
#[derive(Debug)]
pub struct RetryContext {
    pub attempt: u32,
    pub total_elapsed: Duration,
    pub last_error: Option<AiError>,
    pub delay_history: Vec<Duration>,
}

impl Default for RetryContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryContext {
    pub fn new() -> Self {
        Self {
            attempt: 0,
            total_elapsed: Duration::ZERO,
            last_error: None,
            delay_history: Vec::new(),
        }
    }
}

/// Main retry executor
pub struct RetryExecutor {
    config: RetryConfig,
    start_time: Instant,
}

impl RetryExecutor {
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            start_time: Instant::now(),
        }
    }

    /// Execute a function with retry logic
    pub async fn execute<F, Fut, T>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut context = RetryContext::new();
        let mut last_error = None;

        for attempt in 1..=self.config.max_attempts {
            context.attempt = attempt;
            context.total_elapsed = self.start_time.elapsed();

            // Check if we've exceeded maximum total time
            if let Some(max_time) = self.config.max_total_time {
                if context.total_elapsed >= max_time {
                    return Err(last_error.unwrap_or(AiError::TimeoutError {
                        timeout: max_time,
                        retryable: false,
                    }));
                }
            }

            // Execute the operation
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error.clone());
                    context.last_error = Some(error.clone());

                    // Check if we should retry this error
                    if !self.should_retry(&error, &context) {
                        return Err(error);
                    }

                    // Don't delay after the last attempt
                    if attempt < self.config.max_attempts {
                        let delay = self.calculate_delay(&context, &error);
                        context.delay_history.push(delay);

                        // Check total time again after calculating delay
                        if let Some(max_time) = self.config.max_total_time {
                            if context.total_elapsed + delay >= max_time {
                                return Err(error);
                            }
                        }

                        sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| AiError::InternalError {
            message: "Retry loop completed without error".to_string(),
            component: Some("retry".to_string()),
        }))
    }

    /// Determine if an error should be retried
    fn should_retry(&self, error: &AiError, _context: &RetryContext) -> bool {
        match &self.config.retry_condition {
            RetryCondition::Default => error.is_retryable(),
            RetryCondition::Always => true,
            RetryCondition::Never => false,
            RetryCondition::ErrorTypes(types) => {
                let error_type = format!("{:?}", error);
                types.iter().any(|t| error_type.contains(t))
            }
            RetryCondition::Custom => {
                // Default implementation for custom - can be extended
                error.is_retryable()
            }
        }
    }

    /// Calculate the delay before the next retry attempt
    fn calculate_delay(&self, context: &RetryContext, error: &AiError) -> Duration {
        // First check if the error specifies a retry-after delay
        if self.config.respect_retry_after {
            if let Some(retry_after) = error.retry_after() {
                return std::cmp::min(retry_after, self.config.max_delay);
            }
        }

        // Calculate base delay based on backoff strategy
        let base_delay = match &self.config.backoff {
            BackoffStrategy::Fixed => self.config.initial_delay,

            BackoffStrategy::Linear => Duration::from_millis(
                self.config.initial_delay.as_millis() as u64 * context.attempt as u64,
            ),

            BackoffStrategy::Exponential { multiplier } => {
                let delay_ms = self.config.initial_delay.as_millis() as f64
                    * multiplier.powi((context.attempt - 1) as i32);
                Duration::from_millis(delay_ms as u64)
            }

            BackoffStrategy::Custom(delays) => delays
                .get((context.attempt - 1) as usize)
                .copied()
                .unwrap_or(self.config.max_delay),
        };

        // Apply jitter
        let jittered_delay = self.apply_jitter(base_delay, context);

        // Ensure delay doesn't exceed maximum
        std::cmp::min(jittered_delay, self.config.max_delay)
    }

    /// Apply jitter to the delay
    fn apply_jitter(&self, delay: Duration, context: &RetryContext) -> Duration {
        let mut rng = rand::thread_rng();

        match &self.config.jitter {
            JitterStrategy::None => delay,

            JitterStrategy::Full => {
                let jitter_ms = rng.gen_range(0..=delay.as_millis() as u64);
                Duration::from_millis(jitter_ms)
            }

            JitterStrategy::Half => {
                let base_ms = delay.as_millis() as u64 / 2;
                let jitter_ms = base_ms + rng.gen_range(0..=base_ms);
                Duration::from_millis(jitter_ms)
            }

            JitterStrategy::Fixed(jitter_amount) => {
                let jitter_ms = rng.gen_range(0..=jitter_amount.as_millis() as u64);
                delay + Duration::from_millis(jitter_ms)
            }

            JitterStrategy::Decorrelated => {
                // Decorrelated jitter: next_delay = random(base_delay, last_delay * 3)
                let last_delay = context
                    .delay_history
                    .last()
                    .copied()
                    .unwrap_or(self.config.initial_delay);

                let min_delay = delay.as_millis() as u64;
                let max_delay = (last_delay.as_millis() as u64 * 3).max(min_delay);

                let jitter_ms = rng.gen_range(min_delay..=max_delay);
                Duration::from_millis(jitter_ms)
            }
        }
    }
}

/// Builder for creating custom retry configurations
pub struct RetryConfigBuilder {
    config: RetryConfig,
}

impl RetryConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: RetryConfig::default(),
        }
    }

    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.config.max_attempts = attempts;
        self
    }

    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.config.initial_delay = delay;
        self
    }

    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.config.max_delay = delay;
        self
    }

    pub fn exponential_backoff(mut self, multiplier: f64) -> Self {
        self.config.backoff = BackoffStrategy::Exponential { multiplier };
        self
    }

    pub fn linear_backoff(mut self) -> Self {
        self.config.backoff = BackoffStrategy::Linear;
        self
    }

    pub fn fixed_backoff(mut self) -> Self {
        self.config.backoff = BackoffStrategy::Fixed;
        self
    }

    pub fn full_jitter(mut self) -> Self {
        self.config.jitter = JitterStrategy::Full;
        self
    }

    pub fn no_jitter(mut self) -> Self {
        self.config.jitter = JitterStrategy::None;
        self
    }

    pub fn max_total_time(mut self, time: Duration) -> Self {
        self.config.max_total_time = Some(time);
        self
    }

    pub fn respect_retry_after(mut self, respect: bool) -> Self {
        self.config.respect_retry_after = respect;
        self
    }

    pub fn build(self) -> RetryConfig {
        self.config
    }
}

impl Default for RetryConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// CIRCUIT BREAKER

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open the circuit (percentage)
    pub failure_threshold: f64,

    /// Minimum number of requests before considering failure rate
    pub minimum_request_count: u32,

    /// Time window to measure failure rate
    pub measurement_window: Duration,

    /// Time to wait before attempting to close the circuit
    pub recovery_timeout: Duration,

    /// Maximum number of requests to allow in half-open state
    pub half_open_max_requests: u32,

    /// Success threshold to close the circuit in half-open state
    pub success_threshold: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 50.0, // 50% failure rate
            minimum_request_count: 10,
            measurement_window: Duration::from_secs(60),
            recovery_timeout: Duration::from_secs(30),
            half_open_max_requests: 3,
            success_threshold: 60.0, // 60% success rate to close
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    /// Circuit is closed - requests are allowed
    Closed,

    /// Circuit is open - requests are rejected
    Open { opened_at: Instant },

    /// Circuit is half-open - limited requests are allowed to test recovery
    HalfOpen {
        opened_at: Instant,
        attempts: u32,
        successes: u32,
    },
}

/// Request outcome for circuit breaker tracking
#[derive(Debug, Clone)]
enum RequestOutcome {
    Success(Instant),
    Failure(Instant),
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<Mutex<CircuitState>>,
    request_history: Arc<Mutex<VecDeque<RequestOutcome>>>,
    service_name: String,
}

impl CircuitBreaker {
    pub fn new(service_name: impl Into<String>, config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            request_history: Arc::new(Mutex::new(VecDeque::new())),
            service_name: service_name.into(),
        }
    }

    /// Execute a function with circuit breaker protection
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        // Check if circuit allows the request
        if !self.allow_request() {
            let failure_rate = self.calculate_failure_rate();
            return Err(AiError::CircuitBreakerOpen {
                service: self.service_name.clone(),
                failure_rate,
                retry_after: self.config.recovery_timeout,
            });
        }

        let start_time = Instant::now();

        // Execute the operation
        match operation().await {
            Ok(result) => {
                self.record_success(start_time);
                Ok(result)
            }
            Err(error) => {
                self.record_failure(start_time);
                Err(error)
            }
        }
    }

    /// Check if the circuit breaker allows a request
    fn allow_request(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();

        match &*state {
            CircuitState::Closed => true,

            CircuitState::Open { opened_at } => {
                if now.duration_since(*opened_at) >= self.config.recovery_timeout {
                    // Transition to half-open state
                    *state = CircuitState::HalfOpen {
                        opened_at: *opened_at,
                        attempts: 0,
                        successes: 0,
                    };
                    true
                } else {
                    false
                }
            }

            CircuitState::HalfOpen { attempts, .. } => {
                *attempts < self.config.half_open_max_requests
            }
        }
    }

    /// Record a successful request
    fn record_success(&self, request_time: Instant) {
        self.add_request_outcome(RequestOutcome::Success(request_time));

        let mut state = self.state.lock().unwrap();

        if let CircuitState::HalfOpen {
            opened_at,
            attempts,
            successes,
        } = &*state
        {
            let new_attempts = attempts + 1;
            let new_successes = successes + 1;

            let success_rate = (new_successes as f64 / new_attempts as f64) * 100.0;

            if new_attempts >= self.config.half_open_max_requests {
                if success_rate >= self.config.success_threshold {
                    // Close the circuit
                    *state = CircuitState::Closed;
                    // Clear history to start fresh
                    self.request_history.lock().unwrap().clear();
                } else {
                    // Reopen the circuit
                    *state = CircuitState::Open {
                        opened_at: Instant::now(),
                    };
                }
            } else {
                *state = CircuitState::HalfOpen {
                    opened_at: *opened_at,
                    attempts: new_attempts,
                    successes: new_successes,
                };
            }
        }
    }

    /// Record a failed request
    fn record_failure(&self, request_time: Instant) {
        self.add_request_outcome(RequestOutcome::Failure(request_time));

        let mut state = self.state.lock().unwrap();

        match &*state {
            CircuitState::Closed => {
                // Check if we should open the circuit
                if self.should_open_circuit() {
                    *state = CircuitState::Open {
                        opened_at: Instant::now(),
                    };
                }
            }

            CircuitState::HalfOpen { .. } => {
                // Any failure in half-open state reopens the circuit
                *state = CircuitState::Open {
                    opened_at: Instant::now(),
                };
            }

            CircuitState::Open { .. } => {
                // Already open, nothing to do
            }
        }
    }

    /// Add a request outcome to the history
    fn add_request_outcome(&self, outcome: RequestOutcome) {
        let mut history = self.request_history.lock().unwrap();
        let now = Instant::now();

        // Remove old entries outside the measurement window
        while let Some(front) = history.front() {
            let request_time = match front {
                RequestOutcome::Success(time) | RequestOutcome::Failure(time) => *time,
            };

            if now.duration_since(request_time) > self.config.measurement_window {
                history.pop_front();
            } else {
                break;
            }
        }

        // Add the new outcome
        history.push_back(outcome);
    }

    /// Check if the circuit should be opened based on failure rate
    fn should_open_circuit(&self) -> bool {
        let history = self.request_history.lock().unwrap();

        if history.len() < self.config.minimum_request_count as usize {
            return false;
        }

        let failure_rate = self.calculate_failure_rate();
        failure_rate >= self.config.failure_threshold
    }

    /// Calculate the current failure rate
    fn calculate_failure_rate(&self) -> f64 {
        let history = self.request_history.lock().unwrap();

        if history.is_empty() {
            return 0.0;
        }

        let mut failures = 0;
        let total = history.len();

        for outcome in history.iter() {
            if matches!(outcome, RequestOutcome::Failure(_)) {
                failures += 1;
            }
        }

        (failures as f64 / total as f64) * 100.0
    }

    /// Get the current state of the circuit breaker
    pub fn state(&self) -> CircuitState {
        self.state.lock().unwrap().clone()
    }

    /// Get circuit breaker metrics
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        let state = self.state.lock().unwrap().clone();
        let history = self.request_history.lock().unwrap();

        let mut successes = 0;
        let mut failures = 0;

        for outcome in history.iter() {
            match outcome {
                RequestOutcome::Success(_) => successes += 1,
                RequestOutcome::Failure(_) => failures += 1,
            }
        }

        let total_requests = successes + failures;
        let failure_rate = if total_requests > 0 {
            (failures as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        CircuitBreakerMetrics {
            service_name: self.service_name.clone(),
            state: state.clone(),
            total_requests,
            successful_requests: successes,
            failed_requests: failures,
            failure_rate,
            requests_in_window: history.len() as u32,
        }
    }

    /// Reset the circuit breaker to closed state
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Closed;

        let mut history = self.request_history.lock().unwrap();
        history.clear();
    }

    /// Manually open the circuit breaker
    pub fn open(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Open {
            opened_at: Instant::now(),
        };
    }

    /// Manually close the circuit breaker
    pub fn close(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Closed;

        let mut history = self.request_history.lock().unwrap();
        history.clear();
    }
}

/// Circuit breaker metrics
#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    pub service_name: String,
    pub state: CircuitState,
    pub total_requests: u32,
    pub successful_requests: u32,
    pub failed_requests: u32,
    pub failure_rate: f64,
    pub requests_in_window: u32,
}

/// Circuit breaker registry for managing multiple circuit breakers
pub struct CircuitBreakerRegistry {
    breakers: Arc<Mutex<HashMap<String, Arc<CircuitBreaker>>>>,
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: Arc::new(Mutex::new(HashMap::new())),
            default_config,
        }
    }

    /// Get or create a circuit breaker for a service
    pub fn get_or_create(&self, service_name: &str) -> Arc<CircuitBreaker> {
        let mut breakers = self.breakers.lock().unwrap();

        breakers
            .entry(service_name.to_string())
            .or_insert_with(|| {
                Arc::new(CircuitBreaker::new(
                    service_name,
                    self.default_config.clone(),
                ))
            })
            .clone()
    }

    /// Create a circuit breaker with custom config
    pub fn create_with_config(
        &self,
        service_name: &str,
        config: CircuitBreakerConfig,
    ) -> Arc<CircuitBreaker> {
        let mut breakers = self.breakers.lock().unwrap();
        let breaker = Arc::new(CircuitBreaker::new(service_name, config));
        breakers.insert(service_name.to_string(), breaker.clone());
        breaker
    }

    /// Get all circuit breaker metrics
    pub fn get_all_metrics(&self) -> Vec<CircuitBreakerMetrics> {
        let breakers = self.breakers.lock().unwrap();
        breakers.values().map(|breaker| breaker.metrics()).collect()
    }

    /// Reset all circuit breakers
    pub fn reset_all(&self) {
        let breakers = self.breakers.lock().unwrap();
        for breaker in breakers.values() {
            breaker.reset();
        }
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

// RESILIENT PROVIDER

/// A wrapper that adds retry logic and circuit breaker functionality to any provider
pub struct ResilientProvider {
    inner: Arc<dyn crate::CompletionProvider>,
    retry_executor: RetryExecutor,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl ResilientProvider {
    /// Create a new resilient provider with default configuration
    pub fn new(provider: Arc<dyn crate::CompletionProvider>) -> Self {
        let _service_name = format!("provider_{}", provider.name());
        Self::with_config(
            provider,
            RetryConfig::default(),
            CircuitBreakerConfig::default(),
        )
    }

    /// Create a new resilient provider with custom configuration
    pub fn with_config(
        provider: Arc<dyn crate::CompletionProvider>,
        retry_config: RetryConfig,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> Self {
        let _service_name = format!("provider_{}", provider.name());
        let circuit_breaker = Arc::new(CircuitBreaker::new(_service_name, circuit_breaker_config));
        let retry_executor = RetryExecutor::new(retry_config);

        Self {
            inner: provider,
            retry_executor,
            circuit_breaker,
        }
    }

    /// Get the underlying provider
    pub fn inner(&self) -> &Arc<dyn crate::CompletionProvider> {
        &self.inner
    }

    /// Get circuit breaker metrics
    pub fn circuit_breaker_metrics(&self) -> CircuitBreakerMetrics {
        self.circuit_breaker.metrics()
    }

    /// Reset the circuit breaker
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.reset();
    }

    /// Manually open the circuit breaker
    pub fn open_circuit_breaker(&self) {
        self.circuit_breaker.open();
    }

    /// Manually close the circuit breaker
    pub fn close_circuit_breaker(&self) {
        self.circuit_breaker.close();
    }
}

#[async_trait]
impl crate::CompletionProvider for ResilientProvider {
    async fn complete(
        &self,
        request: crate::CompletionRequest,
    ) -> Result<crate::CompletionResponse> {
        let circuit_breaker = self.circuit_breaker.clone();
        let inner = self.inner.clone();
        let request_clone = request.clone();

        // Execute with circuit breaker protection
        circuit_breaker
            .execute(|| {
                let inner = inner.clone();
                let request = request_clone.clone();

                // Execute with retry logic
                self.retry_executor.execute(move || {
                    let inner = inner.clone();
                    let request = request.clone();

                    async move {
                        // Transform provider-specific errors to our enhanced error types
                        inner
                            .complete(request)
                            .await
                            .map_err(|e| enhance_error(e, inner.name()))
                    }
                })
            })
            .await
    }

    async fn complete_stream(
        &self,
        request: crate::CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::StreamChunk>> + Send>>> {
        let circuit_breaker = self.circuit_breaker.clone();
        let inner = self.inner.clone();
        let request_clone = request.clone();

        // For streaming, we apply circuit breaker but not retry logic
        // (since streams are typically long-lived)
        circuit_breaker
            .execute(|| {
                let inner = inner.clone();
                let request = request_clone.clone();

                async move {
                    inner
                        .complete_stream(request)
                        .await
                        .map_err(|e| enhance_error(e, inner.name()))
                }
            })
            .await
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn default_model(&self) -> &'static str {
        self.inner.default_model()
    }

    fn available_models(&self) -> Vec<&'static str> {
        self.inner.available_models()
    }
}

/// Enhance basic errors with more detailed error information
fn enhance_error(error: AiError, provider_name: &str) -> AiError {
    match error {
        // Already enhanced errors - pass through
        e @ AiError::NetworkError { .. }
        | e @ AiError::TimeoutError { .. }
        | e @ AiError::RateLimitExceeded { .. }
        | e @ AiError::InvalidApiKey { .. }
        | e @ AiError::ProviderError { .. } => e,

        // Enhance generic errors with provider context
        AiError::Custom {
            message,
            error_type,
            metadata,
        } => {
            let mut enhanced_metadata = metadata;
            enhanced_metadata.insert("provider".to_string(), provider_name.to_string());

            AiError::Custom {
                message,
                error_type,
                metadata: enhanced_metadata,
            }
        }

        // Transform other errors to provider-specific errors
        e => AiError::ProviderError {
            provider: provider_name.to_string(),
            message: e.to_string(),
            error_code: None,
            retryable: e.is_retryable(),
        },
    }
}

/// Builder for creating resilient providers with custom configuration
pub struct ResilientProviderBuilder {
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
}

impl ResilientProviderBuilder {
    pub fn new() -> Self {
        Self {
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
        }
    }

    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    pub fn circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = config;
        self
    }

    pub fn max_retries(mut self, max_attempts: u32) -> Self {
        self.retry_config.max_attempts = max_attempts;
        self
    }

    pub fn failure_threshold(mut self, threshold: f64) -> Self {
        self.circuit_breaker_config.failure_threshold = threshold;
        self
    }

    pub fn recovery_timeout(mut self, timeout: Duration) -> Self {
        self.circuit_breaker_config.recovery_timeout = timeout;
        self
    }

    pub fn build(self, provider: Arc<dyn crate::CompletionProvider>) -> ResilientProvider {
        ResilientProvider::with_config(provider, self.retry_config, self.circuit_breaker_config)
    }
}

impl Default for ResilientProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenient retry function with default configuration
pub async fn retry_with_default<F, Fut, T>(operation: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let executor = RetryExecutor::new(RetryConfig::default());
    executor.execute(operation).await
}
