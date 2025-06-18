use dotenv::dotenv;
use lib_ai::{
    providers::ReplicateProvider, CompletionProvider, CompletionRequest, Message, MessageContent,
    Role,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Try to create Replicate provider
    let provider = match ReplicateProvider::new(None) {
        Ok(p) => p,
        Err(_) => {
            eprintln!("❌ Error: REPLICATE_API_TOKEN environment variable not set.");
            eprintln!("   Please set it with: export REPLICATE_API_TOKEN=your-api-token");
            eprintln!("   Or create a .env file with REPLICATE_API_TOKEN=your-api-token");
            return Ok(());
        }
    };

    println!("🔄 Replicate Provider Example");
    println!("============================");
    println!("Using models deployed on Replicate\n");

    // Basic completion with Llama 2
    let request = CompletionRequest {
        model: "meta/llama-2-70b-chat".to_string(),
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::text("You are a helpful AI assistant."),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::text(
                    "What is Replicate and how does it help with AI deployment?",
                ),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(300),
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

    println!("📝 Sending request to Replicate (Llama 2 70B)...");
    println!("Note: Replicate predictions may take a moment to start...\n");

    match provider.complete(request.clone()).await {
        Ok(response) => {
            if let Some(text) = response.choices[0].message.content.as_text() {
                println!("📖 Response:");
                println!("{}", text);
            }

            // Token usage
            if let Some(usage) = response.usage {
                println!("\n📊 Token Usage:");
                println!("  Prompt tokens: {}", usage.prompt_tokens);
                println!("  Completion tokens: {}", usage.completion_tokens);
                println!(
                    "  Total tokens: {}",
                    usage.total_tokens
                );
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            eprintln!("\nMake sure:");
            eprintln!("1. Your REPLICATE_API_TOKEN is valid");
            eprintln!("2. The model 'meta/llama-2-70b-chat' is available");
            eprintln!("3. You have sufficient credits on Replicate");
        }
    }

    // Try a smaller model
    println!("\n🚀 Trying a smaller model (Llama 2 7B):");
    let mut small_request = request;
    small_request.model = "meta/llama-2-7b-chat".to_string();
    small_request.messages[1].content = MessageContent::text("Write a haiku about cloud computing");
    small_request.max_tokens = Some(100);

    match provider.complete(small_request).await {
        Ok(response) => {
            if let Some(text) = response.choices[0].message.content.as_text() {
                println!("\n📖 Haiku:");
                println!("{}", text);
            }
        }
        Err(e) => {
            eprintln!("\n❌ Error with smaller model: {}", e);
        }
    }

    // Note about streaming
    println!("\n📝 Note about streaming:");
    println!("Replicate supports streaming, but it works differently than other providers.");
    println!("The initial response may take longer as the model needs to be loaded.");

    // Available models info
    println!("\n📋 Popular Replicate models:");
    println!("  - meta/llama-2-7b-chat");
    println!("  - meta/llama-2-13b-chat");
    println!("  - meta/llama-2-70b-chat");
    println!("  - replicate/vicuna-13b");
    println!("  - stability-ai/stablelm-tuned-alpha-7b");

    println!("\n💡 Tips:");
    println!("  - Check https://replicate.com/explore for available models");
    println!("  - Models may need to 'cold start' if not recently used");
    println!("  - Pricing varies by model and usage");

    println!("\n✅ Replicate example completed!");

    Ok(())
}
