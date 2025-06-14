use lib_ai::{
    error::{BackoffStrategy, JitterStrategy},
    error::{CircuitBreakerConfig, ResilientProvider, ResilientProviderBuilder, RetryConfig},
    providers::OpenAIProvider,
    CompletionProvider, CompletionRequest, Message, MessageContent, Role,
};
use std::env;
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Get API key from environment
    let api_key =
        env::var("OPENAI_API_KEY").expect("Please set OPENAI_API_KEY environment variable");

    println!("🛡️  Resilient Provider Example");
    println!("============================");
    println!("Demonstrating retry logic and circuit breakers\n");

    // Create base provider
    let base_provider = Arc::new(OpenAIProvider::new(api_key));

    // Example 1: Default resilient provider
    println!("1️⃣  Default Resilient Provider");
    let resilient = ResilientProvider::new(base_provider.clone());

    let request = create_test_request();
    match resilient.complete(request.clone()).await {
        Ok(response) => {
            if let Some(text) = response.choices[0].message.content.as_text() {
                println!("Response: {}", text);
            }
        }
        Err(e) => println!("Error (will retry automatically): {}", e),
    }

    // Example 2: Custom retry configuration
    println!("\n2️⃣  Custom Retry Configuration");
    let retry_config = RetryConfig {
        max_attempts: 5,
        initial_delay: Duration::from_millis(500),
        max_delay: Duration::from_secs(30),
        backoff: BackoffStrategy::Exponential { multiplier: 2.0 },
        jitter: JitterStrategy::Full,
        respect_retry_after: true,
        max_total_time: Some(Duration::from_secs(120)),
        retry_condition: lib_ai::error::RetryCondition::Default,
    };

    let circuit_config = CircuitBreakerConfig {
        failure_threshold: 50.0,
        minimum_request_count: 5,
        measurement_window: Duration::from_secs(60),
        recovery_timeout: Duration::from_secs(30),
        half_open_max_requests: 3,
        success_threshold: 60.0,
    };

    let custom_resilient =
        ResilientProvider::with_config(base_provider.clone(), retry_config, circuit_config);

    // Example 3: Using the builder
    println!("\n3️⃣  Builder Pattern");
    let builder_resilient = ResilientProviderBuilder::new()
        .max_retries(3)
        .failure_threshold(75.0)
        .recovery_timeout(Duration::from_secs(45))
        .build(base_provider.clone());

    // Test with a request
    match builder_resilient.complete(request.clone()).await {
        Ok(response) => {
            if let Some(text) = response.choices[0].message.content.as_text() {
                println!("Response: {}", text.chars().take(100).collect::<String>());
            }
        }
        Err(e) => println!("Final error after retries: {}", e),
    }

    // Example 4: Circuit breaker metrics
    println!("\n4️⃣  Circuit Breaker Metrics");
    let metrics = custom_resilient.circuit_breaker_metrics();
    println!("Service: {}", metrics.service_name);
    println!("State: {:?}", metrics.state);
    println!("Total requests: {}", metrics.total_requests);
    println!("Failed requests: {}", metrics.failed_requests);
    println!("Failure rate: {:.2}%", metrics.failure_rate);

    // Example 5: Simulating failures
    println!("\n5️⃣  Simulating Failures");

    // Create a provider that will fail (invalid URL)
    let failing_provider = Arc::new(OpenAIProvider::with_base_url(
        "invalid-key".to_string(),
        "http://localhost:9999".to_string(), // Non-existent endpoint
    ));

    let resilient_failing = ResilientProviderBuilder::new()
        .max_retries(2)
        .build(failing_provider);

    println!("Attempting request to failing endpoint...");
    match resilient_failing.complete(request.clone()).await {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => {
            println!("Expected failure: {}", e);
            println!("Is retryable: {}", e.is_retryable());
            if let Some(delay) = e.retry_after() {
                println!("Retry after: {:?}", delay);
            }
        }
    }

    // Example 6: Fallback pattern
    println!("\n6️⃣  Fallback Pattern");
    let primary = ResilientProvider::new(base_provider.clone());
    let fallback = ResilientProvider::new(base_provider.clone());

    let result = match primary.complete(request.clone()).await {
        Ok(response) => {
            println!("Primary provider succeeded");
            response
        }
        Err(e) => {
            println!("Primary failed: {}, trying fallback...", e);
            fallback.complete(request).await?
        }
    };

    if let Some(text) = result.choices[0].message.content.as_text() {
        println!(
            "Final response: {}",
            text.chars().take(100).collect::<String>()
        );
    }

    println!("\n✅ Resilient provider examples completed!");

    Ok(())
}

fn create_test_request() -> CompletionRequest {
    CompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::text("Say 'Hello, resilient world!' in a creative way"),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: Some(0.7),
        max_tokens: Some(50),
        stream: Some(false),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        json_schema: None,
    }
}
