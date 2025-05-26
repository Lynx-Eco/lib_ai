use lib_ai::{providers::*, CompletionProvider, CompletionRequest, Message, Role, MessageContent};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    
    let provider = OpenAIProvider::new(api_key);
    
    let request = CompletionRequest {
        model: provider.default_model().to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful assistant."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text("What is 2+2?"),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(150),
        stream: Some(false),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        json_schema: None,
    };
    
    let response = provider.complete(request).await?;
    
    println!("Model: {}", response.model);
    for choice in response.choices {
        if let Some(text) = choice.message.content.as_text() {
            println!("Response: {}", text);
        }
    }
    
    if let Some(usage) = response.usage {
        println!("Tokens used: {}", usage.total_tokens);
    }
    
    Ok(())
}
