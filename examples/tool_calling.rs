use lib_ai::{
    providers::*, CompletionProvider, CompletionRequest, Message, Role, MessageContent,
    Tool, ToolType, ToolFunction, ToolChoice,
};
use serde_json::json;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let provider = OpenAIProvider::new(api_key);
    
    // Define a weather tool
    let weather_tool = Tool {
        r#type: ToolType::Function,
        function: ToolFunction {
            name: "get_weather".to_string(),
            description: Some("Get the current weather in a given location".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location", "unit"]
            }),
        },
    };
    
    // Make a request with tools
    let request = CompletionRequest {
        model: provider.default_model().to_string(),
        messages: vec![
            Message {
                role: Role::User,
                content: MessageContent::text("What's the weather like in San Francisco?"),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(150),
        stream: Some(false),
        tools: Some(vec![weather_tool]),
        tool_choice: Some(ToolChoice::String("auto".to_string())),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        response_format: None,
        json_schema: None,
    };
    
    let response = provider.complete(request).await?;
    
    println!("Model: {}", response.model);
    for choice in response.choices {
        if let Some(text) = choice.message.content.as_text() {
            println!("Response: {}", text);
        }
        
        if let Some(tool_calls) = choice.message.tool_calls {
            for tool_call in tool_calls {
                println!("Tool Call ID: {}", tool_call.id);
                println!("Function: {}", tool_call.function.name);
                println!("Arguments: {}", tool_call.function.arguments);
            }
        }
    }
    
    Ok(())
}