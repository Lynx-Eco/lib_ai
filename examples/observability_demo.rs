use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;
use lib_ai::{
    agent::AgentBuilder, CompletionProvider,
    observability::{
        MetricsCollector, AgentTracer, CostTracker, TelemetryExporter,
        telemetry::{TelemetryConfig, ExporterConfig, ExporterType},
        tracing::TracingConfig,
    },
    agent::tools::{CalculatorTool, WebFetchTool},
};

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

    async fn complete(&self, request: lib_ai::CompletionRequest) -> Result<lib_ai::CompletionResponse, lib_ai::AiError> {
        use lib_ai::{CompletionResponse, Choice, Message, MessageContent, Role, Usage};
        
        // Simulate a response
        Ok(CompletionResponse {
            id: "mock-response".to_string(),
            model: request.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::text("This is a mock response for observability testing"),
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
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<lib_ai::StreamChunk, lib_ai::AiError>> + Send>>,
        lib_ai::AiError,
    > {
        use futures::stream;
        Ok(Box::pin(stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 AI Agent Observability Demo");
    println!("================================");

    // Initialize observability components
    let metrics_collector = Arc::new(MetricsCollector::new());
    let tracer = Arc::new(AgentTracer::new(TracingConfig {
        enabled: true,
        sample_rate: 1.0, // Sample all traces for demo
        max_traces: 1000,
        max_spans_per_trace: 100,
        export_interval: Duration::from_secs(30),
    }));
    // Create two cost trackers due to type mismatch between builder and telemetry
    let cost_tracker_for_agent = Arc::new(std::sync::RwLock::new(CostTracker::new()));
    let cost_tracker_for_telemetry = Arc::new(tokio::sync::RwLock::new(CostTracker::new()));
    
    // Create telemetry exporter
    let telemetry_config = TelemetryConfig {
        enabled: true,
        export_interval: Duration::from_secs(30),
        exporters: vec![
            ExporterConfig {
                name: "console".to_string(),
                enabled: true,
                exporter_type: ExporterType::Console,
                endpoint: None,
                headers: HashMap::new(),
            },
            ExporterConfig {
                name: "file".to_string(),
                enabled: true,
                exporter_type: ExporterType::File { path: "telemetry.log".to_string() },
                endpoint: None,
                headers: HashMap::new(),
            },
        ],
        batch_size: 100,
        max_queue_size: 1000,
    };
    let telemetry_exporter = Arc::new(TelemetryExporter::new(
        telemetry_config,
        metrics_collector.clone(),
        tracer.clone(),
        cost_tracker_for_telemetry.clone(),
    ));

    // Create agent with observability
    let mut agent = AgentBuilder::new()
        .provider(MockProvider)
        .prompt("You are a helpful assistant with calculator and web browsing capabilities")
        .model("mock-model")
        .temperature(0.7)
        .tool("calculator", CalculatorTool)
        .tool("web_fetch", WebFetchTool::new())
        .with_observability(
            metrics_collector.clone(),
            tracer.clone(),
            cost_tracker_for_agent.clone(),
            telemetry_exporter.clone(),
        )
        .build()?;

    println!("🚀 Agent created with full observability");
    println!("Agent ID: {}", agent.agent_id());
    
    // Perform some operations to generate metrics and traces
    println!("\n📊 Executing agent tasks...");
    
    let tasks = vec![
        "What is 42 + 58?",
        "Calculate the square root of 144",
        "What is the meaning of life?",
        "Explain quantum computing in simple terms",
    ];
    
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
        println!("  Success Rate: {:.2}%", 
            if metrics.total_requests > 0 {
                (metrics.successful_requests as f64 / metrics.total_requests as f64) * 100.0
            } else {
                0.0
            }
        );
        println!("  Average Response Time: {:.2}ms", metrics.average_response_time.as_millis());
        println!("  Total Tokens: {}", metrics.total_tokens.total());
        println!("  Total Cost: ${:.6}", metrics.total_cost);
        
        if !metrics.tool_usage.is_empty() {
            println!("  Tool Executions:");
            for (tool_name, tool_metrics) in &metrics.tool_usage {
                println!("    {}: {} executions, {:.2}ms avg", 
                    tool_name, 
                    tool_metrics.executions,
                    tool_metrics.average_duration.as_millis()
                );
            }
        }
    }
    
    // Display global metrics
    let global_metrics = metrics_collector.get_global_metrics();
        println!("\n🌍 Global Metrics:");
        println!("  Total Agents: {}", global_metrics.total_agents);
        println!("  Total Requests: {}", global_metrics.total_requests);
        println!("  Total Tokens: {}", global_metrics.total_tokens.total());
        println!("  Total Cost: ${:.6}", global_metrics.total_cost);

    // Display cost breakdown
    println!("\n💰 Cost Breakdown:");
    println!("==================");
    
    if let Ok(cost_tracker_guard) = cost_tracker_for_agent.read() {
        let report = cost_tracker_guard.generate_report();
        println!("Total Cost: ${:.6}", report.total_cost);
        
        for provider in &report.providers {
            println!("  {}: ${:.6}", provider.provider_name, provider.total_cost);
            for model in &provider.models {
                println!("    {}: ${:.6}", model.model_name, model.total_cost);
            }
        }
    }

    // Display trace information
    println!("\n🔍 Trace Information:");
    println!("=====================");
    
    let all_traces = tracer.get_all_traces();
    println!("Total Traces: {}", all_traces.len());
    
    for (i, (trace_id, events)) in all_traces.iter().take(3).enumerate() {
        if let Some(first_event) = events.first() {
            let duration = events.iter()
                .filter_map(|e| e.duration)
                .map(|d| d.as_millis())
                .sum::<u128>();
            
            println!("Trace {}: {} - {} ({}ms)", 
                i + 1, 
                trace_id,
                first_event.operation_name,
                duration
            );
        }
    }

    // Export telemetry data
    println!("\n📤 Exporting Telemetry Data...");
    
    // Export telemetry (telemetry exporter runs in background)
    if let Err(e) = telemetry_exporter.export_now().await {
        println!("❌ Failed to export telemetry: {}", e);
    } else {
        println!("✅ Telemetry data exported successfully");
    }

    println!("\n🎉 Observability demo completed!");
    println!("Check 'telemetry.log' for exported data");

    Ok(())
}