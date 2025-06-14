use lib_ai::agent::{AgentBuilder, StructuredOutput, StructuredProvider, TypedAgentBuilder};
use lib_ai::providers::OpenAIProvider;
use lib_ai::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

// Define our structured response types
#[derive(Debug, Serialize, Deserialize, Default)]
struct WeatherInfo {
    location: String,
    temperature: f32,
    conditions: String,
    humidity: u8,
    wind_speed: f32,
}

// Implement StructuredProvider for WeatherInfo
impl StructuredProvider for WeatherInfo {
    fn schema() -> JsonSchema {
        JsonSchema {
            name: "WeatherInfo".to_string(),
            description: Some("Weather information for a location".to_string()),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "The location for the weather" },
                    "temperature": { "type": "number", "description": "Temperature in Celsius" },
                    "conditions": { "type": "string", "description": "Weather conditions (e.g., sunny, cloudy, rainy)" },
                    "humidity": { "type": "integer", "description": "Humidity percentage (0-100)" },
                    "wind_speed": { "type": "number", "description": "Wind speed in km/h" }
                },
                "required": ["location", "temperature", "conditions", "humidity", "wind_speed"]
            }),
            strict: Some(true),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CodeAnalysis {
    language: String,
    complexity: String,
    suggestions: Vec<String>,
    has_bugs: bool,
    estimated_time: f32,
}

impl StructuredProvider for CodeAnalysis {
    fn schema() -> JsonSchema {
        JsonSchema {
            name: "CodeAnalysis".to_string(),
            description: Some("Code analysis results".to_string()),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "language": { "type": "string", "description": "Programming language detected" },
                    "complexity": { "type": "string", "enum": ["low", "medium", "high"], "description": "Code complexity level" },
                    "suggestions": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of improvement suggestions"
                    },
                    "has_bugs": { "type": "boolean", "description": "Whether potential bugs were detected" },
                    "estimated_time": { "type": "number", "description": "Estimated time to implement in hours" }
                },
                "required": ["language", "complexity", "suggestions", "has_bugs", "estimated_time"]
            }),
            strict: Some(true),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    // Get API key
    let api_key =
        env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable must be set");

    println!("Structured Agent Examples");
    println!("========================\n");

    // Example 1: TypedAgent for Weather
    println!("1. TypedAgent Example - Weather Information:");
    println!("-------------------------------------------");

    let mut weather_agent = TypedAgentBuilder::<WeatherInfo>::new()
        .provider(OpenAIProvider::new(api_key.clone()))
        .prompt("You are a weather information assistant. Always respond with accurate weather data in the specified format.")
        .model("gpt-4o-mini")
        .temperature(0.3)
        .build()
        .expect("Failed to build weather agent");

    let weather_query = "What's the weather like in Paris, France?";
    println!("Query: {}", weather_query);

    match weather_agent.chat(weather_query).await {
        Ok(weather) => {
            println!("Response: {:?}", weather);
            println!("  Location: {}", weather.location);
            println!("  Temperature: {}°C", weather.temperature);
            println!("  Conditions: {}", weather.conditions);
            println!("  Humidity: {}%", weather.humidity);
            println!("  Wind Speed: {} km/h", weather.wind_speed);
        }
        Err(e) => println!("Error: {}", e),
    }

    // Example 2: Regular Agent with Structured Output
    println!("\n\n2. Regular Agent with Structured Output - Code Analysis:");
    println!("--------------------------------------------------------");

    let mut code_agent = AgentBuilder::new()
        .provider(OpenAIProvider::new(api_key))
        .prompt("You are a code analysis assistant. Analyze the provided code and return structured insights.")
        .model("gpt-4o-mini")
        .temperature(0.2)
        .build()
        .expect("Failed to build code agent");

    let code_sample = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(result)
"#;

    let analysis_query = format!(
        "Analyze this code and provide a detailed analysis:\n```python\n{}\n```",
        code_sample
    );
    println!("Query: Analyzing Python fibonacci code...");

    // Use the StructuredOutput trait
    match code_agent.chat_typed::<CodeAnalysis>(&analysis_query).await {
        Ok(analysis) => {
            println!("Analysis Result:");
            println!("  Language: {}", analysis.language);
            println!("  Complexity: {}", analysis.complexity);
            println!("  Has Bugs: {}", analysis.has_bugs);
            println!("  Estimated Time: {} hours", analysis.estimated_time);
            println!("  Suggestions:");
            for (i, suggestion) in analysis.suggestions.iter().enumerate() {
                println!("    {}. {}", i + 1, suggestion);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // Example 3: Dynamic structured responses
    println!("\n\n3. Dynamic Structured Response:");
    println!("-------------------------------");

    #[derive(Debug, Serialize, Deserialize)]
    struct TaskList {
        tasks: Vec<Task>,
        total_time: f32,
        priority: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Task {
        name: String,
        duration: f32,
        difficulty: String,
    }

    impl StructuredProvider for TaskList {
        fn schema() -> JsonSchema {
            JsonSchema {
                name: "TaskList".to_string(),
                description: Some("Task breakdown with time estimates".to_string()),
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "duration": { "type": "number" },
                                    "difficulty": { "type": "string", "enum": ["easy", "medium", "hard"] }
                                },
                                "required": ["name", "duration", "difficulty"]
                            }
                        },
                        "total_time": { "type": "number" },
                        "priority": { "type": "string", "enum": ["low", "medium", "high", "urgent"] }
                    },
                    "required": ["tasks", "total_time", "priority"]
                }),
                strict: Some(true),
            }
        }
    }

    let task_query = "Break down the process of building a simple web application into tasks";
    println!("Query: {}", task_query);

    match code_agent.execute_typed::<TaskList>(task_query).await {
        Ok(task_list) => {
            println!("Task Breakdown:");
            println!("  Priority: {}", task_list.priority);
            println!("  Total Time: {} hours", task_list.total_time);
            println!("  Tasks:");
            for task in &task_list.tasks {
                println!(
                    "    - {} ({} hours, {})",
                    task.name, task.duration, task.difficulty
                );
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}
