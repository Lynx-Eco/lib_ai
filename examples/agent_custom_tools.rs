use lib_ai::{
    agent::{AgentBuilder, ToolExecutor, ToolResult, tools::FunctionTool},
    providers::OpenAIProvider,
    ToolFunction,
};
use async_trait::async_trait;
use serde_json::json;
use tokio;

// Custom tool that gets the current time
struct TimeTool;

#[async_trait]
impl ToolExecutor for TimeTool {
    async fn execute(&self, _arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        use chrono::Local;
        
        let now = Local::now();
        let time_str = now.format("%Y-%m-%d %H:%M:%S %Z").to_string();
        
        Ok(ToolResult::Success(json!({
            "time": time_str,
            "timezone": "local"
        })))
    }
    
    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "get_current_time".to_string(),
            description: Some("Get the current date and time".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }
}

// Custom tool that rolls dice
struct DiceTool;

#[async_trait]
impl ToolExecutor for DiceTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let args: serde_json::Value = serde_json::from_str(arguments)?;
        
        let num_dice = args["num_dice"].as_u64().unwrap_or(1) as u32;
        let sides = args["sides"].as_u64().unwrap_or(6) as u32;
        
        if num_dice == 0 || num_dice > 100 {
            return Ok(ToolResult::Error("Number of dice must be between 1 and 100".to_string()));
        }
        
        if sides < 2 || sides > 1000 {
            return Ok(ToolResult::Error("Number of sides must be between 2 and 1000".to_string()));
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut results = Vec::new();
        let mut total = 0;
        
        for _ in 0..num_dice {
            let roll = rng.gen_range(1..=sides);
            results.push(roll);
            total += roll;
        }
        
        let response = json!({
            "rolls": results,
            "total": total,
            "dice": format!("{}d{}", num_dice, sides)
        });
        
        Ok(ToolResult::Success(response))
    }
    
    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "roll_dice".to_string(),
            description: Some("Roll dice and get the results".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "num_dice": {
                        "type": "integer",
                        "description": "Number of dice to roll (1-100)",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "sides": {
                        "type": "integer",
                        "description": "Number of sides on each die (2-1000)",
                        "minimum": 2,
                        "maximum": 1000
                    }
                },
                "required": ["num_dice", "sides"]
            }),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")?;
    
    // Create a provider
    let provider = OpenAIProvider::new(api_key);
    
    // Create a simple uppercase function using FunctionTool
    let uppercase_tool = FunctionTool::new(
        "uppercase".to_string(),
        "Convert text to uppercase".to_string(),
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert to uppercase"
                }
            },
            "required": ["text"]
        }),
        |args| {
            let parsed: serde_json::Value = serde_json::from_str(args)?;
            let text = parsed["text"].as_str().ok_or("Missing text")?;
            Ok(text.to_uppercase())
        },
    );
    
    // Build an agent with custom tools
    let agent = AgentBuilder::new()
        .provider(provider)
        .prompt("You are a helpful assistant with access to various tools.")
        .model("gpt-4o-mini")
        .temperature(0.7)
        .tool("current_time", TimeTool)
        .tool("roll_dice", DiceTool)
        .tool("uppercase", uppercase_tool)
        .build()?;
    
    let mut agent = agent;
    
    // Example interactions
    println!("=== Agent with Custom Tools ===\n");
    
    // Test time tool
    println!("User: What time is it?");
    let response = agent.execute("What time is it?").await?;
    println!("Assistant: {}\n", response);
    
    // Test dice tool
    println!("User: Roll 3 six-sided dice for me");
    let response = agent.execute("Roll 3 six-sided dice for me").await?;
    println!("Assistant: {}\n", response);
    
    // Test uppercase tool
    println!("User: Convert 'hello world' to uppercase");
    let response = agent.execute("Convert 'hello world' to uppercase").await?;
    println!("Assistant: {}\n", response);
    
    // Test multiple tools in one request
    println!("User: What's the current time, and roll 2d20 for me");
    let response = agent.execute("What's the current time, and roll 2d20 for me").await?;
    println!("Assistant: {}", response);
    
    Ok(())
}