use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

use crate::agent::tools::{ToolExecutor, ToolResult};
use crate::ToolFunction;

/// Code execution tool for running code in various languages
pub struct CodeExecutorTool {
    /// Maximum execution time
    timeout_secs: u64,
    /// Maximum output size in bytes
    max_output_size: usize,
    /// Allowed languages
    allowed_languages: Vec<String>,
}

impl Default for CodeExecutorTool {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeExecutorTool {
    /// Create a new code executor tool
    pub fn new() -> Self {
        Self {
            timeout_secs: 30,
            max_output_size: 1024 * 1024, // 1MB
            allowed_languages: vec![
                "python".to_string(),
                "javascript".to_string(),
                "bash".to_string(),
                "sh".to_string(),
            ],
        }
    }
    
    /// Set execution timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
    
    /// Set maximum output size
    pub fn with_max_output_size(mut self, size: usize) -> Self {
        self.max_output_size = size;
        self
    }
    
    /// Add allowed language
    pub fn add_allowed_language(mut self, language: impl Into<String>) -> Self {
        self.allowed_languages.push(language.into());
        self
    }
    
    /// Get command for language
    fn get_command(&self, language: &str) -> Option<(&str, Vec<&str>)> {
        match language.to_lowercase().as_str() {
            "python" | "python3" => Some(("python3", vec!["-c"])),
            "javascript" | "js" | "node" => Some(("node", vec!["-e"])),
            "bash" => Some(("bash", vec!["-c"])),
            "sh" => Some(("sh", vec!["-c"])),
            "ruby" => Some(("ruby", vec!["-e"])),
            "perl" => Some(("perl", vec!["-e"])),
            _ => None,
        }
    }
}

use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct CodeExecutionRequest {
    /// Programming language
    language: String,
    /// Code to execute
    code: String,
    /// Optional stdin input
    stdin: Option<String>,
    /// Environment variables
    env: Option<HashMap<String, String>>,
}

#[async_trait]
impl ToolExecutor for CodeExecutorTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let input: Value = serde_json::from_str(arguments)?;
        let request: CodeExecutionRequest = serde_json::from_value(input)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid input: {}", e))) as Box<dyn std::error::Error>)?;
        
        // Check if language is allowed
        if !self.allowed_languages.contains(&request.language.to_lowercase()) {
            return Ok(ToolResult::Error(format!(
                "Language '{}' is not allowed. Allowed languages: {:?}",
                request.language,
                self.allowed_languages
            )));
        }
        
        // Get command for language
        let (cmd, args) = self.get_command(&request.language)
            .ok_or_else(|| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Unsupported language: {}", request.language))) as Box<dyn std::error::Error>)?;
        
        // Create command
        let mut command = Command::new(cmd);
        command.args(args);
        command.arg(&request.code);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        
        // Add environment variables if provided
        if let Some(env_vars) = request.env {
            for (key, value) in env_vars {
                command.env(key, value);
            }
        }
        
        // Spawn process
        let mut child = command.spawn()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        
        // Write stdin if provided
        if let Some(stdin_data) = request.stdin {
            if let Some(mut stdin) = child.stdin.take() {
                use tokio::io::AsyncWriteExt;
                stdin.write_all(stdin_data.as_bytes()).await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            }
        }
        
        // Wait for completion with timeout
        let output = timeout(
            Duration::from_secs(self.timeout_secs),
            child.wait_with_output()
        ).await
            .map_err(|_| Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, format!("Execution timed out after {} seconds", self.timeout_secs))) as Box<dyn std::error::Error>)?
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        
        // Check output size
        let stdout_len = output.stdout.len();
        let stderr_len = output.stderr.len();
        
        if stdout_len + stderr_len > self.max_output_size {
            return Ok(ToolResult::Error(format!(
                "Output too large: {} bytes (max: {} bytes)",
                stdout_len + stderr_len,
                self.max_output_size
            )));
        }
        
        // Convert output to strings
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        Ok(ToolResult::Success(serde_json::json!({
            "success": output.status.success(),
            "exit_code": output.status.code(),
            "stdout": stdout,
            "stderr": stderr,
            "language": request.language,
        })))
    }
    
    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "code_executor".to_string(),
            description: Some("Execute code in various programming languages".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": self.allowed_languages,
                        "description": "Programming language"
                    },
                    "code": {
                        "type": "string",
                        "description": "Code to execute"
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Optional standard input"
                    },
                    "env": {
                        "type": "object",
                        "description": "Optional environment variables",
                        "additionalProperties": { "type": "string" }
                    }
                },
                "required": ["language", "code"]
            }),
        }
    }
}

