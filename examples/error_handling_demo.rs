use std::sync::Arc;
use std::time::Duration;
use std::sync::atomic::{AtomicU32, Ordering};
use lib_ai::{
    CompletionProvider, CompletionRequest, CompletionResponse, StreamChunk,
    AiError, ResilientProviderBuilder,
    RetryConfigBuilder, CircuitBreakerConfig,
    agent::AgentBuilder,
    agent::tools::CalculatorTool,
};
use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;

/// Mock provider that simulates various error conditions
struct UnreliableProvider {
    request_count: AtomicU32,
    failure_pattern: FailurePattern,
}

#[derive(Debug, Clone)]
enum FailurePattern {
    /// Always succeeds
    AlwaysSucceed,
    
    /// Fails for first N requests, then succeeds
    FailThenSucceed { fail_count: u32 },
    
    /// Alternates between success and failure
    Alternating,
    
    /// Always fails with a specific error
    AlwaysFail { error: AiError },
    
    /// Fails with rate limiting
    RateLimit { after_requests: u32 },
    
    /// Simulates network timeouts
    Timeout { probability: f32 },
}

impl UnreliableProvider {
    fn new(pattern: FailurePattern) -> Self {
        Self {
            request_count: AtomicU32::new(0),
            failure_pattern: pattern,
        }
    }
    
    fn should_fail(&self) -> Option<AiError> {
        let count = self.request_count.fetch_add(1, Ordering::SeqCst);
        
        match &self.failure_pattern {
            FailurePattern::AlwaysSucceed => None,
            
            FailurePattern::FailThenSucceed { fail_count } => {
                if count < *fail_count {
                    Some(AiError::NetworkError {
                        message: "Simulated network failure".to_string(),
                        retryable: true,
                        status_code: Some(500),
                    })
                } else {
                    None
                }
            }
            
            FailurePattern::Alternating => {
                if count % 2 == 0 {
                    Some(AiError::ServiceUnavailable {
                        provider: "unreliable".to_string(),
                        retry_after: Some(Duration::from_millis(100)),
                    })
                } else {
                    None
                }
            }
            
            FailurePattern::AlwaysFail { error } => Some(error.clone()),
            
            FailurePattern::RateLimit { after_requests } => {
                if count >= *after_requests {
                    Some(AiError::RateLimitExceeded {
                        retry_after: Some(Duration::from_millis(500)),
                        daily_limit: Some(1000),
                        requests_remaining: Some(0),
                    })
                } else {
                    None
                }
            }
            
            FailurePattern::Timeout { probability } => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                if rng.gen::<f32>() < *probability {
                    Some(AiError::TimeoutError {
                        timeout: Duration::from_secs(30),
                        retryable: true,
                    })
                } else {
                    None
                }
            }
        }
    }
}

#[async_trait]
impl CompletionProvider for UnreliableProvider {
    async fn complete(&self, request: CompletionRequest) -> lib_ai::Result<CompletionResponse> {
        // Simulate some processing time
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        if let Some(error) = self.should_fail() {
            return Err(error);
        }
        
        // Return a successful response
        Ok(CompletionResponse {
            id: format!("req_{}", self.request_count.load(Ordering::SeqCst)),
            model: request.model,
            choices: vec![],
            usage: Some(lib_ai::Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        })
    }

    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> lib_ai::Result<Pin<Box<dyn Stream<Item = lib_ai::Result<StreamChunk>> + Send>>> {
        use futures::stream;
        Ok(Box::pin(stream::empty()))
    }

    fn name(&self) -> &'static str {
        "unreliable"
    }

    fn default_model(&self) -> &'static str {
        "test-model"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec!["test-model"]
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🔧 AI Error Handling & Retry Logic Demo");
    println!("=======================================");

    // Demo 1: Basic retry with exponential backoff
    println!("\n1️⃣ Basic Retry with Exponential Backoff");
    println!("----------------------------------------");
    
    let unreliable_provider = Arc::new(UnreliableProvider::new(
        FailurePattern::FailThenSucceed { fail_count: 2 }
    ));
    
    let retry_config = RetryConfigBuilder::new()
        .max_attempts(5)
        .initial_delay(Duration::from_millis(100))
        .exponential_backoff(2.0)
        .build();
    
    let resilient_provider = ResilientProviderBuilder::new()
        .retry_config(retry_config)
        .failure_threshold(80.0) // High threshold for circuit breaker
        .build(unreliable_provider);
    
    let request = create_test_request();
    
    let start_time = std::time::Instant::now();
    match resilient_provider.complete(request).await {
        Ok(response) => {
            println!("✅ Success after retries! Response ID: {}", response.id);
            println!("⏱️  Total time: {:?}", start_time.elapsed());
        }
        Err(e) => {
            println!("❌ Failed after retries: {}", e);
        }
    }

    // Demo 2: Circuit breaker in action
    println!("\n2️⃣ Circuit Breaker Demo");
    println!("------------------------");
    
    let always_fail_provider = Arc::new(UnreliableProvider::new(
        FailurePattern::AlwaysFail {
            error: AiError::NetworkError {
                message: "Persistent network failure".to_string(),
                retryable: true,
                status_code: Some(503),
            }
        }
    ));
    
    let circuit_breaker_config = CircuitBreakerConfig {
        failure_threshold: 50.0,
        minimum_request_count: 3,
        measurement_window: Duration::from_secs(60),
        recovery_timeout: Duration::from_secs(2),
        half_open_max_requests: 2,
        success_threshold: 60.0,
    };
    
    let resilient_provider = ResilientProviderBuilder::new()
        .max_retries(1) // Don't retry much to test circuit breaker
        .circuit_breaker_config(circuit_breaker_config)
        .build(always_fail_provider);
    
    println!("Making requests to trigger circuit breaker...");
    
    for i in 1..=6 {
        let request = create_test_request();
        let start_time = std::time::Instant::now();
        
        match resilient_provider.complete(request).await {
            Ok(_) => println!("Request {}: ✅ Success", i),
            Err(e) => {
                let elapsed = start_time.elapsed();
                match &e {
                    AiError::CircuitBreakerOpen { service, failure_rate, .. } => {
                        println!("Request {}: 🚫 Circuit breaker open for {} (failure rate: {:.1}%) - {:?}", 
                            i, service, failure_rate, elapsed);
                    }
                    _ => {
                        println!("Request {}: ❌ Failed: {} - {:?}", i, e, elapsed);
                    }
                }
            }
        }
        
        // Show circuit breaker state
        let metrics = resilient_provider.circuit_breaker_metrics();
        println!("  Circuit state: {:?}, Failure rate: {:.1}%", 
            metrics.state, metrics.failure_rate);
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Demo 3: Rate limiting with respect for retry-after
    println!("\n3️⃣ Rate Limiting Demo");
    println!("---------------------");
    
    let rate_limit_provider = Arc::new(UnreliableProvider::new(
        FailurePattern::RateLimit { after_requests: 2 }
    ));
    
    let resilient_provider = ResilientProviderBuilder::new()
        .max_retries(3)
        .build(rate_limit_provider);
    
    for i in 1..=4 {
        let request = create_test_request();
        let start_time = std::time::Instant::now();
        
        match resilient_provider.complete(request).await {
            Ok(response) => {
                println!("Request {}: ✅ Success - {} ({:?})", 
                    i, response.id, start_time.elapsed());
            }
            Err(e) => {
                println!("Request {}: ❌ {}", i, e);
                if let Some(retry_after) = e.retry_after() {
                    println!("  Retry after: {:?}", retry_after);
                }
            }
        }
    }

    // Demo 4: Agent integration with error handling
    println!("\n4️⃣ Agent with Error Handling");
    println!("-----------------------------");
    
    let flaky_provider = Arc::new(UnreliableProvider::new(
        FailurePattern::Timeout { probability: 0.3 }
    ));
    
    let resilient_provider = ResilientProviderBuilder::new()
        .max_retries(3)
        // Note: initial_delay is configured via retry_config
        .build(flaky_provider);
    
    let mut agent = AgentBuilder::new()
        .provider_arc(Arc::new(resilient_provider))
        .prompt("You are a helpful assistant")
        .tool("calculator", CalculatorTool)
        .build().map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error>)?;
    
    println!("Testing agent with flaky provider...");
    
    let tasks = vec![
        "Hello! How are you?",
        "What is 2 + 2?",
        "Tell me a joke",
    ];
    
    for task in tasks {
        println!("\nTask: {}", task);
        let start_time = std::time::Instant::now();
        
        match agent.execute(task).await {
            Ok(response) => {
                println!("✅ Response: {} (took {:?})", 
                    response, start_time.elapsed());
            }
            Err(e) => {
                println!("❌ Error: {} (took {:?})", 
                    e, start_time.elapsed());
                // AgentError wraps AiError - need to extract it
                if let lib_ai::agent::AgentError::ProviderError(ai_error) = &e {
                    println!("  Error severity: {}", ai_error.severity().as_str());
                    println!("  Is retryable: {}", ai_error.is_retryable());
                    if let Some(provider) = ai_error.provider() {
                        println!("  Provider: {}", provider);
                    }
                }
            }
        }
    }

    // Demo 5: Error classification and handling
    println!("\n5️⃣ Error Classification Demo");
    println!("----------------------------");
    
    let errors = vec![
        AiError::InvalidApiKey { provider: "openai".to_string() },
        AiError::RateLimitExceeded { 
            retry_after: Some(Duration::from_secs(60)),
            daily_limit: Some(1000),
            requests_remaining: Some(0),
        },
        AiError::NetworkError { 
            message: "Connection timeout".to_string(),
            retryable: true,
            status_code: Some(504),
        },
        AiError::ContentFiltered { 
            reason: "Inappropriate content detected".to_string(),
            category: Some("violence".to_string()),
        },
        AiError::ToolExecutionError { 
            tool_name: "calculator".to_string(),
            message: "Division by zero".to_string(),
            retryable: false,
        },
    ];
    
    for error in errors {
        println!("\nError: {}", error);
        println!("  Severity: {}", error.severity().as_str());
        println!("  Retryable: {}", error.is_retryable());
        if let Some(retry_after) = error.retry_after() {
            println!("  Retry after: {:?}", retry_after);
        }
        if let Some(provider) = error.provider() {
            println!("  Provider: {}", provider);
        }
    }

    // Demo 6: Custom error with metadata
    println!("\n6️⃣ Custom Error with Metadata");
    println!("------------------------------");
    
    let custom_error = AiError::custom("Custom business logic error", "business_logic")
        .with_metadata("retryable", "true")
        .with_metadata("error_code", "BL001")
        .with_metadata("component", "payment_processor");
    
    println!("Custom error: {}", custom_error);
    println!("  Retryable: {}", custom_error.is_retryable());
    
    if let AiError::Custom { metadata, .. } = &custom_error {
        println!("  Metadata:");
        for (key, value) in metadata {
            println!("    {}: {}", key, value);
        }
    }

    println!("\n🎉 Error handling demo completed!");
    
    Ok(())
}

fn create_test_request() -> CompletionRequest {
    CompletionRequest {
        model: "test-model".to_string(),
        messages: vec![],
        temperature: None,
        max_tokens: None,
        stream: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        json_schema: None,
    }
}