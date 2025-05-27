use lib_ai::{
    providers::*, CompletionProvider, CompletionRequest, Message, Role, MessageContent,
    ResponseFormat, ResponseFormatType,
};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let provider = OpenAIProvider::new(api_key);
    
    // Request JSON output
    let request = CompletionRequest {
        model: provider.default_model().to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful assistant that outputs JSON."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text(
                    "Extract the following information from this text and return as JSON: \
                    'John Doe is 30 years old and lives in New York. He works as a software engineer.'"
                ),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(150),
        stream: Some(false),
        response_format: Some(ResponseFormat {
            r#type: ResponseFormatType::JsonObject,
        }),
        tools: None,
        tool_choice: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        json_schema: None,
    };
    
    let response = provider.complete(request).await?;
    
    println!("Model: {}", response.model);
    for choice in response.choices {
        if let Some(text) = choice.message.content.as_text() {
            println!("JSON Response: {}", text);
            
            // Parse the JSON to verify it's valid
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                println!("Parsed JSON: {:#?}", json);
            }
        }
    }
    
    Ok(())
}