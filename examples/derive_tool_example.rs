use lib_ai::{
    agent::{AgentBuilder, ToolExecutor, ToolResult},
    providers::OpenAIProvider,
};
use serde::{Deserialize, Serialize};
use std::error::Error;

// Example of a tool that could use the AiTool derive macro
#[derive(Debug, Serialize, Deserialize)]
struct CalculatorInput {
    operation: String,
    a: f64,
    b: f64,
}

// Manual implementation of ToolExecutor (what AiTool derive would generate)
struct CalculatorTool;

#[async_trait::async_trait]
impl ToolExecutor for CalculatorTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let input: CalculatorInput = serde_json::from_str(arguments)?;

        let result = match input.operation.as_str() {
            "add" => input.a + input.b,
            "subtract" => input.a - input.b,
            "multiply" => input.a * input.b,
            "divide" => {
                if input.b == 0.0 {
                    return Ok(ToolResult::Error("Division by zero".to_string()));
                }
                input.a / input.b
            }
            _ => {
                return Ok(ToolResult::Error(format!(
                    "Unknown operation: {}",
                    input.operation
                )))
            }
        };

        Ok(ToolResult::Success(serde_json::json!({
            "result": result,
            "operation": input.operation,
            "a": input.a,
            "b": input.b
        })))
    }

    fn definition(&self) -> lib_ai::ToolFunction {
        lib_ai::ToolFunction {
            name: "calculator".to_string(),
            description: Some("Perform mathematical calculations".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Mathematical operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First operand"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second operand"
                    }
                },
                "required": ["operation", "a", "b"]
            }),
        }
    }
}

// Another tool example
struct TimezoneTool;

#[async_trait::async_trait]
impl ToolExecutor for TimezoneTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        use chrono::{DateTime, FixedOffset, Utc};

        #[derive(Deserialize)]
        struct TimezoneInput {
            timezone_offset: i32, // Offset in hours from UTC
        }

        let input: TimezoneInput = serde_json::from_str(arguments)?;

        let now_utc: DateTime<Utc> = Utc::now();
        let offset =
            FixedOffset::east_opt(input.timezone_offset * 3600).ok_or("Invalid timezone offset")?;
        let now_local = now_utc.with_timezone(&offset);

        Ok(ToolResult::Success(serde_json::json!({
            "utc_time": now_utc.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            "local_time": now_local.format("%Y-%m-%d %H:%M:%S %:z").to_string(),
            "timezone_offset": input.timezone_offset,
        })))
    }

    fn definition(&self) -> lib_ai::ToolFunction {
        lib_ai::ToolFunction {
            name: "get_time_in_timezone".to_string(),
            description: Some("Get current time in a specific timezone".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "timezone_offset": {
                        "type": "integer",
                        "description": "Timezone offset in hours from UTC (e.g., -5 for EST, 8 for CST)",
                        "minimum": -12,
                        "maximum": 14
                    }
                },
                "required": ["timezone_offset"]
            }),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let provider = OpenAIProvider::new(api_key);

    // Create an agent with multiple tools
    let mut agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are a helpful assistant with calculation and timezone capabilities.")
        .tool("calculator", CalculatorTool)
        .tool("get_time_in_timezone", TimezoneTool)
        .build()?;

    // Example 1: Simple calculation
    println!("=== Calculation Example ===");
    let response = agent.execute("What is 42 multiplied by 17?").await?;
    println!("Response: {}", response);

    // Example 2: Timezone query
    println!("\n=== Timezone Example ===");
    let response = agent.execute("What time is it in Tokyo (UTC+9)?").await?;
    println!("Response: {}", response);

    // Example 3: Combined query
    println!("\n=== Combined Example ===");
    let response = agent
        .execute("Calculate 24 * 7 and tell me what time it is in New York (UTC-5)")
        .await?;
    println!("Response: {}", response);

    Ok(())
}
