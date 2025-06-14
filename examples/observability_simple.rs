use lib_ai::{
    agent::tools::CalculatorTool,
    agent::AgentBuilder,
    observability::{tracing::TracingConfig, AgentTracer, CostTracker, MetricsCollector},
    CompletionProvider,
};
use std::sync::Arc;
use std::time::Duration;

struct MockProvider;

#[async_trait::async_trait]
impl CompletionProvider for MockProvider {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn default_model(&self) -> &'static str {
        "mock-model"
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec!["mock-model"]
    }

    async fn complete(
        &self,
        request: lib_ai::CompletionRequest,
    ) -> Result<lib_ai::CompletionResponse, lib_ai::AiError> {
        use lib_ai::{Choice, CompletionResponse, Message, MessageContent, Role, Usage};

        // Simulate a response
        Ok(CompletionResponse {
            id: "mock-response".to_string(),
            model: request.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::text(
                        "This is a mock response for observability testing",
                    ),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 50,
                completion_tokens: 20,
                total_tokens: 70,
            }),
        })
    }

    async fn complete_stream(
        &self,
        _request: lib_ai::CompletionRequest,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<lib_ai::StreamChunk, lib_ai::AiError>> + Send>,
        >,
        lib_ai::AiError,
    > {
        use futures::stream;
        Ok(Box::pin(stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 AI Agent Observability Demo (Simplified)");
    println!("============================================");

    // Initialize observability components
    let metrics_collector = Arc::new(MetricsCollector::new());
    let tracer = Arc::new(AgentTracer::new(TracingConfig {
        enabled: true,
        sample_rate: 1.0,
        max_traces: 100,
        max_spans_per_trace: 10,
        export_interval: Duration::from_secs(60),
    }));
    let cost_tracker = Arc::new(std::sync::RwLock::new(CostTracker::new()));

    // Create agent with observability
    let mut agent = AgentBuilder::new()
        .provider(MockProvider)
        .prompt("You are a helpful assistant")
        .model("mock-model")
        .temperature(0.7)
        .tool("calculator", CalculatorTool)
        .metrics_collector(metrics_collector.clone())
        .tracer(tracer.clone())
        .cost_tracker(cost_tracker.clone())
        .build()?;

    println!("🚀 Agent created with observability");
    println!("Agent ID: {}", agent.agent_id());

    // Perform some operations to generate metrics and traces
    println!("\n📊 Executing agent tasks...");

    let tasks = vec!["Hello, how are you?", "What is 2 + 2?", "Tell me about AI"];

    for (i, task) in tasks.iter().enumerate() {
        println!("\nTask {}: {}", i + 1, task);

        match agent.execute(task).await {
            Ok(response) => {
                println!("✅ Response: {}", response);
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }

        // Small delay between requests
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Display collected metrics
    println!("\n📈 Collected Metrics:");
    println!("===================");

    if let Some(metrics) = metrics_collector.get_agent_metrics(agent.agent_id()) {
        println!("Agent: {}", agent.agent_id());
        println!("  Total Requests: {}", metrics.total_requests);
        println!("  Successful Requests: {}", metrics.successful_requests);
        println!("  Failed Requests: {}", metrics.failed_requests);
        println!(
            "  Success Rate: {:.2}%",
            if metrics.total_requests > 0 {
                (metrics.successful_requests as f64 / metrics.total_requests as f64) * 100.0
            } else {
                0.0
            }
        );
        println!(
            "  Average Response Time: {:.2}ms",
            metrics.average_response_time.as_millis()
        );
        println!(
            "  Total Input Tokens: {}",
            metrics.total_tokens.input_tokens
        );
        println!(
            "  Total Output Tokens: {}",
            metrics.total_tokens.output_tokens
        );
        println!("  Total Cost: ${:.6}", metrics.total_cost);
    }

    // Display global metrics
    let global_metrics = metrics_collector.get_global_metrics();
    println!("\n🌍 Global Metrics:");
    println!("  Total Agents: {}", global_metrics.total_agents);
    println!("  Total Requests: {}", global_metrics.total_requests);
    println!(
        "  Total Input Tokens: {}",
        global_metrics.total_tokens.input_tokens
    );
    println!(
        "  Total Output Tokens: {}",
        global_metrics.total_tokens.output_tokens
    );
    println!("  Total Cost: ${:.6}", global_metrics.total_cost);

    // Display cost breakdown
    println!("\n💰 Cost Breakdown:");
    println!("==================");

    if let Ok(cost_tracker_guard) = cost_tracker.read() {
        let report = cost_tracker_guard.generate_report();
        println!("Total Cost: ${:.6}", report.total_cost);

        for provider in &report.providers {
            println!("  {}: ${:.6}", provider.provider_name, provider.total_cost);
            for model in &provider.models {
                println!(
                    "    {}: ${:.6} ({} requests)",
                    model.model_name, model.total_cost, model.requests
                );
            }
        }
    }

    // Display trace information
    println!("\n🔍 Trace Information:");
    println!("=====================");

    let all_traces = tracer.get_all_traces();
    println!("Total Traces: {}", all_traces.len());

    for (i, (trace_id, events)) in all_traces.iter().take(3).enumerate() {
        println!("Trace {}: {} ({} events)", i + 1, trace_id, events.len());

        for event in events.iter().take(2) {
            println!(
                "  - {}: {}ms",
                event.operation_name,
                event.duration.map_or(0, |d| d.as_millis())
            );
        }
    }

    println!("\n🎉 Observability demo completed!");

    Ok(())
}
