#[cfg(feature = "derive")]
use lib_ai::Structured;
use lib_ai::{
    agent::{StructuredProvider, TypedAgentBuilder},
    providers::OpenAIProvider,
};
use lib_ai_derive::Structured;
use serde::{Deserialize, Serialize};
use std::error::Error;

// Example 1: Structured Output with Structured derive
#[derive(Debug, Serialize, Deserialize, Structured)]
struct WeatherResponse {
    #[schema(description = "Current temperature in Celsius")]
    temperature: f32,

    #[schema(description = "Weather condition (e.g., sunny, cloudy, rainy)")]
    condition: String,

    #[schema(description = "Humidity percentage (0-100)")]
    humidity: u8,

    #[schema(description = "Wind speed in km/h")]
    wind_speed: Option<f32>,
}

// Example 2: Enum with Structured
#[derive(Debug, Serialize, Deserialize, Structured)]
enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Serialize, Deserialize, Structured)]
struct TaskResponse {
    #[schema(description = "Unique task identifier")]
    id: String,

    #[schema(description = "Task description")]
    description: String,

    #[schema(description = "Current status of the task")]
    status: TaskStatus,

    #[schema(description = "Progress percentage (0-100)")]
    progress: Option<u8>,
}

// Example 3: Complex nested structure
#[derive(Debug, Serialize, Deserialize, Structured)]
struct Address {
    #[schema(description = "Street address")]
    street: String,

    #[schema(description = "City name")]
    city: String,

    #[schema(description = "Postal code")]
    postal_code: String,

    #[schema(description = "Country")]
    country: String,
}

#[derive(Debug, Serialize, Deserialize, Structured)]
struct UserProfile {
    #[schema(description = "User's full name")]
    name: String,

    #[schema(description = "User's email address")]
    email: String,

    #[schema(description = "User's age")]
    age: Option<u32>,

    #[schema(description = "User's address")]
    address: Address,

    #[schema(description = "User's interests")]
    interests: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Example 1: Using structured output with derived schema
    println!("=== Structured Output Example ===");
    let mut weather_agent = TypedAgentBuilder::<WeatherResponse>::new()
        .provider(OpenAIProvider::new(api_key.clone()))
        .prompt(
            "You are a weather information assistant. Always respond with current weather data.",
        )
        .build()?;

    let weather = weather_agent
        .execute("What's the weather like in London?")
        .await?;
    println!("Weather: {:?}", weather);

    // Example 2: Task management with enum
    println!("\n=== Task Management Example ===");
    let mut task_agent = TypedAgentBuilder::<TaskResponse>::new()
        .provider(OpenAIProvider::new(api_key.clone()))
        .prompt("You are a task management assistant. Create tasks based on user requests.")
        .build()?;

    let task = task_agent
        .execute("Create a task to review the quarterly report")
        .await?;
    println!("Task: {:?}", task);

    // Example 3: User profile extraction
    println!("\n=== User Profile Example ===");
    let mut profile_agent = TypedAgentBuilder::<UserProfile>::new()
        .provider(OpenAIProvider::new(api_key))
        .prompt("You are a profile extraction assistant. Extract user information from text.")
        .build()?;

    let profile = profile_agent.execute(
        "John Doe is a 30-year-old software engineer living at 123 Main St, San Francisco, CA 94105, USA. \
         His email is john@example.com and he enjoys hiking, photography, and cooking."
    ).await?;
    println!("Profile: {:?}", profile);

    // Example 4: Print the generated schemas
    println!("\n=== Generated Schemas ===");
    println!("WeatherResponse schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&WeatherResponse::schema().schema)?
    );

    println!("\nTaskResponse schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&TaskResponse::schema().schema)?
    );

    println!("\nTaskStatus schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&TaskStatus::schema().schema)?
    );

    println!("\nUserProfile schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&UserProfile::schema().schema)?
    );

    Ok(())
}
