use lib_ai::{
    agent::{AgentBuilder, InMemoryStore, ToolRegistry, ToolExecutor, ToolResult, tools::CalculatorTool},
    providers::OpenAIProvider,
};
use async_trait::async_trait;
use mockito::{Server, ServerGuard};

async fn create_mock_server() -> ServerGuard {
    Server::new_async().await
}

#[tokio::test]
async fn test_agent_builder() {
    // Test that agent builder creates agents correctly
    let provider = OpenAIProvider::new("test-key".to_string());
    
    let agent = AgentBuilder::new()
        .provider(provider)
        .prompt("Test prompt")
        .model("gpt-3.5-turbo")
        .temperature(0.5)
        .max_tokens(100)
        .build();
    
    assert!(agent.is_ok());
}

#[tokio::test]
async fn test_agent_with_tools() {
    let mut server = create_mock_server().await;
    
    // Mock response with tool call
    let tool_response = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": "{\"operation\": \"multiply\", \"a\": 42, \"b\": 17}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }"#;
    
    // Mock the follow-up response after tool execution
    let final_response = r#"{
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1677652290,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "42 * 17 = 714"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    }"#;
    
    let _mock1 = server.mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(tool_response)
        .create_async()
        .await;
    
    let _mock2 = server.mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(final_response)
        .create_async()
        .await;
    
    let provider = OpenAIProvider::with_base_url("test-key".to_string(), server.url());
    
    let mut agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are a calculator assistant")
        .tool("calculator", CalculatorTool)
        .build()
        .unwrap();
    
    let result = agent.execute("What is 42 * 17?").await.unwrap();
    assert!(result.contains("714"));
}

#[tokio::test]
async fn test_agent_memory() {
    let mut server = create_mock_server().await;
    
    // First interaction
    let _mock1 = server.mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{
            "id": "1",
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "I'll remember that."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#)
        .create_async()
        .await;
    
    // Second interaction (should include memory context)
    let _mock2 = server.mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{
            "id": "2",
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Your favorite color is blue."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }"#)
        .create_async()
        .await;
    
    let provider = OpenAIProvider::with_base_url("test-key".to_string(), server.url());
    
    let mut agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You have perfect memory")
        .memory(InMemoryStore::new(10))
        .build()
        .unwrap();
    
    // First interaction
    agent.execute("My favorite color is blue").await.unwrap();
    
    // Second interaction should use memory
    let response = agent.execute("What's my favorite color?").await.unwrap();
    assert!(response.contains("blue"));
}

#[tokio::test]
async fn test_context_management() {
    let mut server = create_mock_server().await;
    
    let _mock = server.mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{
            "id": "1",
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#)
        .expect(2) // Expect two calls
        .create_async()
        .await;
    
    let provider = OpenAIProvider::with_base_url("test-key".to_string(), server.url());
    
    let mut agent = AgentBuilder::new()
        .provider(provider)
        .prompt("Test agent")
        .build()
        .unwrap();
    
    // Execute once
    agent.execute("Hello").await.unwrap();
    
    // Clear context
    agent.clear_context();
    
    // Context should be reset
    assert_eq!(agent.context().len(), 1); // Only system message
    
    // Execute again
    agent.execute("Hello again").await.unwrap();
}

// Custom test tool
struct TestTool {
    return_value: String,
}

#[async_trait]
impl ToolExecutor for TestTool {
    async fn execute(&self, _args: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        Ok(ToolResult::Success(serde_json::json!({
            "result": self.return_value.clone()
        })))
    }
    
    fn definition(&self) -> lib_ai::ToolFunction {
        lib_ai::ToolFunction {
            name: "test_tool".to_string(),
            description: Some("A test tool".to_string()),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }
    }
}

#[tokio::test]
async fn test_tool_registry() {
    let mut registry = ToolRegistry::new();
    
    registry.register("calc", CalculatorTool);
    registry.register("test", TestTool { return_value: "test result".to_string() });
    
    assert_eq!(registry.len(), 2);
    assert!(registry.contains("calc"));
    assert!(registry.contains("test"));
    
    let tools = registry.to_tools();
    assert_eq!(tools.len(), 2);
}

#[tokio::test]
async fn test_calculator_tool() {
    let calc = CalculatorTool;
    
    // Test addition
    let result = calc.execute(r#"{"operation": "add", "a": 5, "b": 3}"#).await.unwrap();
    match result {
        ToolResult::Success(val) => {
            assert_eq!(val["result"], 8.0);
            assert_eq!(val["operation"], "add");
        },
        ToolResult::Error(e) => panic!("Unexpected error: {}", e),
    }
    
    // Test division
    let result = calc.execute(r#"{"operation": "divide", "a": 10, "b": 2}"#).await.unwrap();
    match result {
        ToolResult::Success(val) => {
            assert_eq!(val["result"], 5.0);
            assert_eq!(val["operation"], "divide");
        },
        ToolResult::Error(e) => panic!("Unexpected error: {}", e),
    }
    
    // Test division by zero
    let result = calc.execute(r#"{"operation": "divide", "a": 10, "b": 0}"#).await.unwrap();
    match result {
        ToolResult::Error(e) => assert_eq!(e, "Division by zero"),
        ToolResult::Success(_) => panic!("Expected error for division by zero"),
    }
}