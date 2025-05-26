use lib_ai::{
    providers::*, CompletionProvider, CompletionRequest, Message, Role, MessageContent,
    ContentPart, ImageUrl,
};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let provider = OpenAIProvider::new(api_key);
    
    // Create a multimodal message with text and image
    let request = CompletionRequest {
        model: "gpt-4o".to_string(), // Use a vision-capable model
        messages: vec![
            Message {
                role: Role::User,
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "What's in this image?".to_string(),
                    },
                    ContentPart::Image {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("high".to_string()),
                        },
                    },
                ]),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(300),
        stream: Some(false),
        tools: None,
        tool_choice: None,
        response_format: None,
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
            println!("Response: {}", text);
        }
    }
    
    Ok(())
}