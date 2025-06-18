use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Tool, ToolFunction, ToolType};

/// Result of a tool execution
#[derive(Debug, Clone)]
pub enum ToolResult {
    Success(Value),
    Error(String),
}

/// Trait for implementing tool executors
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute the tool with the given arguments
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>>;

    /// Get the tool definition
    fn definition(&self) -> ToolFunction;
}

/// Registry for managing tools
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn ToolExecutor>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register<S: Into<String>, E: ToolExecutor + 'static>(&mut self, name: S, executor: E) {
        self.tools.insert(name.into(), Arc::new(executor));
    }

    /// Get a tool executor by name
    pub fn get_executor(&self, name: &str) -> Option<Arc<dyn ToolExecutor>> {
        self.tools.get(name).cloned()
    }

    /// Convert to a vector of Tool definitions for API calls
    pub fn to_tools(&self) -> Vec<Tool> {
        self.tools
            .iter()
            .map(|(name, executor)| {
                let mut definition = executor.definition();
                definition.name = name.clone(); // Ensure name matches registry key

                Tool {
                    r#type: ToolType::Function,
                    function: definition,
                }
            })
            .collect()
    }

    /// Get all tool names
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a tool exists
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Remove a tool
    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn ToolExecutor>> {
        self.tools.remove(name)
    }

    /// Clear all tools
    pub fn clear(&mut self) {
        self.tools.clear();
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Example tool implementations

/// A simple calculator tool
pub struct CalculatorTool;

#[async_trait]
impl ToolExecutor for CalculatorTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let args: Value = serde_json::from_str(arguments)?;

        let operation = args["operation"].as_str().ok_or("Missing operation")?;

        let a = args["a"].as_f64().ok_or("Missing or invalid 'a'")?;

        let b = args["b"].as_f64().ok_or("Missing or invalid 'b'")?;

        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Ok(ToolResult::Error("Division by zero".to_string()));
                }
                a / b
            }
            _ => {
                return Ok(ToolResult::Error(format!(
                    "Unknown operation: {}",
                    operation
                )))
            }
        };

        Ok(ToolResult::Success(serde_json::json!({
            "result": result,
            "operation": operation,
            "a": a,
            "b": b
        })))
    }

    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "calculator".to_string(),
            description: Some("Perform basic arithmetic operations".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "The first operand"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second operand"
                    }
                },
                "required": ["operation", "a", "b"]
            }),
        }
    }
}

/// A tool that can fetch data from a URL
pub struct WebFetchTool {
    client: reqwest::Client,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl ToolExecutor for WebFetchTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let args: Value = serde_json::from_str(arguments)?;

        let url = args["url"].as_str().ok_or("Missing URL")?;

        match self.client.get(url).send().await {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    let text = response.text().await?;
                    // Truncate if too long
                    let truncated = if text.len() > 1000 {
                        format!("{}... (truncated)", &text[..1000])
                    } else {
                        text
                    };
                    Ok(ToolResult::Success(serde_json::json!({
                        "url": url,
                        "status": status.as_u16(),
                        "content": truncated
                    })))
                } else {
                    Ok(ToolResult::Error(format!("HTTP {}", status)))
                }
            }
            Err(e) => Ok(ToolResult::Error(format!("Request failed: {}", e))),
        }
    }

    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "web_fetch".to_string(),
            description: Some("Fetch content from a URL".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }),
        }
    }
}

/// A simple key-value store tool
pub struct KeyValueStoreTool {
    store: Arc<tokio::sync::Mutex<HashMap<String, String>>>,
}

impl Default for KeyValueStoreTool {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyValueStoreTool {
    pub fn new() -> Self {
        Self {
            store: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl ToolExecutor for KeyValueStoreTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let args: Value = serde_json::from_str(arguments)?;

        let action = args["action"].as_str().ok_or("Missing action")?;

        let key = args["key"].as_str().ok_or("Missing key")?;

        let mut store = self.store.lock().await;

        match action {
            "get" => match store.get(key) {
                Some(value) => Ok(ToolResult::Success(serde_json::json!({
                    "key": key,
                    "value": value,
                    "found": true
                }))),
                None => Ok(ToolResult::Success(serde_json::json!({
                    "key": key,
                    "value": null,
                    "found": false
                }))),
            },
            "set" => {
                let value = args["value"]
                    .as_str()
                    .ok_or("Missing value for set action")?;
                store.insert(key.to_string(), value.to_string());
                Ok(ToolResult::Success(serde_json::json!({
                    "key": key,
                    "value": value,
                    "action": "set",
                    "success": true
                })))
            }
            "delete" => {
                store.remove(key);
                Ok(ToolResult::Success(serde_json::json!({
                    "key": key,
                    "action": "delete",
                    "success": true
                })))
            }
            "list" => {
                let keys: Vec<&str> = store.keys().map(|k| k.as_str()).collect();
                Ok(ToolResult::Success(serde_json::json!({
                    "keys": keys,
                    "count": keys.len()
                })))
            }
            _ => Ok(ToolResult::Error(format!("Unknown action: {}", action))),
        }
    }

    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "key_value_store".to_string(),
            description: Some("A simple key-value store".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "set", "delete", "list"],
                        "description": "The action to perform"
                    },
                    "key": {
                        "type": "string",
                        "description": "The key to operate on"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to set (required for 'set' action)"
                    }
                },
                "required": ["action", "key"]
            }),
        }
    }
}

/// Create a simple function tool from a closure
pub struct FunctionTool<F> {
    name: String,
    description: String,
    parameters: Value,
    func: F,
}

impl<F> FunctionTool<F>
where
    F: Fn(&str) -> Result<Value, Box<dyn std::error::Error>> + Send + Sync,
{
    pub fn new(name: String, description: String, parameters: Value, func: F) -> Self {
        Self {
            name,
            description,
            parameters,
            func,
        }
    }
}

#[async_trait]
impl<F> ToolExecutor for FunctionTool<F>
where
    F: Fn(&str) -> Result<Value, Box<dyn std::error::Error>> + Send + Sync,
{
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        match (self.func)(arguments) {
            Ok(result) => Ok(ToolResult::Success(result)),
            Err(e) => Ok(ToolResult::Error(e.to_string())),
        }
    }

    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: self.name.clone(),
            description: Some(self.description.clone()),
            parameters: self.parameters.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculator_tool() {
        let calc = CalculatorTool;

        let result = calc
            .execute(r#"{"operation": "add", "a": 5, "b": 3}"#)
            .await
            .unwrap();
        match result {
            ToolResult::Success(val) => {
                assert_eq!(val["result"], 8.0);
                assert_eq!(val["operation"], "add");
            }
            ToolResult::Error(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();

        registry.register("calculator", CalculatorTool);

        assert!(registry.contains("calculator"));
        assert_eq!(registry.len(), 1);

        let tools = registry.to_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "calculator");
    }
}
