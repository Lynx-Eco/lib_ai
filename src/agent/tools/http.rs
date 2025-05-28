use async_trait::async_trait;
use reqwest::{Client, Method, header::HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use std::collections::HashMap;

use crate::agent::tools::{ToolExecutor, ToolResult};
use crate::ToolFunction;

/// HTTP client tool for making API requests
pub struct HttpTool {
    client: Client,
    /// Maximum response size in bytes
    max_response_size: usize,
    /// Request timeout
    timeout: Duration,
    /// Allowed domains (if empty, all domains are allowed)
    allowed_domains: Vec<String>,
    /// Default headers to include in all requests
    default_headers: HeaderMap,
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpTool {
    /// Create a new HTTP tool
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        
        Self {
            client,
            max_response_size: 1024 * 1024, // 1MB default
            timeout: Duration::from_secs(30),
            allowed_domains: Vec::new(),
            default_headers: HeaderMap::new(),
        }
    }
    
    /// Set maximum response size
    pub fn with_max_response_size(mut self, size: usize) -> Self {
        self.max_response_size = size;
        self
    }
    
    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap();
        self
    }
    
    /// Add allowed domain
    pub fn add_allowed_domain(mut self, domain: impl Into<String>) -> Self {
        self.allowed_domains.push(domain.into());
        self
    }
    
    /// Add default header
    pub fn add_default_header(mut self, key: &str, value: &str) -> Self {
        use reqwest::header::{HeaderName, HeaderValue};
        self.default_headers.insert(
            HeaderName::from_bytes(key.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap()
        );
        self
    }
    
    /// Check if domain is allowed
    fn is_domain_allowed(&self, url: &str) -> bool {
        if self.allowed_domains.is_empty() {
            return true;
        }
        
        if let Ok(parsed) = url::Url::parse(url) {
            if let Some(host) = parsed.host_str() {
                return self.allowed_domains.iter().any(|domain| {
                    host == domain || host.ends_with(&format!(".{}", domain))
                });
            }
        }
        
        false
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct HttpRequest {
    /// HTTP method (GET, POST, PUT, DELETE, etc.)
    method: String,
    /// URL to request
    url: String,
    /// Optional headers
    headers: Option<HashMap<String, String>>,
    /// Optional request body (for POST, PUT, etc.)
    body: Option<Value>,
    /// Optional query parameters
    params: Option<HashMap<String, String>>,
}

#[async_trait]
impl ToolExecutor for HttpTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let input: Value = serde_json::from_str(arguments)?;
        let request: HttpRequest = serde_json::from_value(input)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid input: {}", e))) as Box<dyn std::error::Error>)?;
        
        // Validate domain
        if !self.is_domain_allowed(&request.url) {
            return Ok(ToolResult::Error(format!("Domain not allowed: {}", request.url)));
        }
        
        // Parse method
        let method = Method::from_bytes(request.method.to_uppercase().as_bytes())
            .map_err(|_| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Invalid HTTP method: {}", request.method))) as Box<dyn std::error::Error>)?;
        
        // Build request
        let mut req = self.client.request(method, &request.url);
        
        // Add default headers
        for (key, value) in self.default_headers.iter() {
            req = req.header(key.clone(), value.clone());
        }
        
        // Add custom headers
        if let Some(headers) = request.headers {
            for (key, value) in headers {
                req = req.header(key, value);
            }
        }
        
        // Add query parameters
        if let Some(params) = request.params {
            req = req.query(&params);
        }
        
        // Add body
        if let Some(body) = request.body {
            req = req.json(&body);
        }
        
        // Send request
        let response = req.send().await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        
        // Get response info
        let status = response.status();
        let headers = response.headers().clone();
        
        // Check content length
        if let Some(content_length) = response.content_length() {
            if content_length > self.max_response_size as u64 {
                return Ok(ToolResult::Error(format!(
                    "Response too large: {} bytes (max: {} bytes)",
                    content_length,
                    self.max_response_size
                )));
            }
        }
        
        // Read response body
        let body_bytes = response.bytes().await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        
        if body_bytes.len() > self.max_response_size {
            return Ok(ToolResult::Error(format!(
                "Response too large: {} bytes (max: {} bytes)",
                body_bytes.len(),
                self.max_response_size
            )));
        }
        
        // Try to parse as JSON, otherwise return as text
        let body = if let Ok(json) = serde_json::from_slice::<Value>(&body_bytes) {
            json
        } else {
            Value::String(String::from_utf8_lossy(&body_bytes).to_string())
        };
        
        // Convert headers to JSON
        let mut response_headers = HashMap::new();
        for (key, value) in headers.iter() {
            response_headers.insert(
                key.to_string(),
                value.to_str().unwrap_or("").to_string()
            );
        }
        
        Ok(ToolResult::Success(serde_json::json!({
            "status": status.as_u16(),
            "status_text": status.canonical_reason().unwrap_or(""),
            "headers": response_headers,
            "body": body
        })))
    }
    
    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "http".to_string(),
            description: Some("Make HTTP requests to APIs and web services".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                        "description": "HTTP method"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to request"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers",
                        "additionalProperties": { "type": "string" }
                    },
                    "body": {
                        "description": "Optional request body (JSON)"
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional query parameters",
                        "additionalProperties": { "type": "string" }
                    }
                },
                "required": ["method", "url"]
            }),
        }
    }
}

