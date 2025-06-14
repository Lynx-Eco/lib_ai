use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

use crate::agent::tools::{ToolExecutor, ToolResult};
use crate::ToolFunction;

/// File system tool for reading and writing files
pub struct FileSystemTool {
    /// Base directory for file operations (sandboxing)
    base_dir: PathBuf,
    /// Whether to allow write operations
    allow_write: bool,
    /// Maximum file size to read (in bytes)
    max_file_size: usize,
}

impl FileSystemTool {
    /// Create a new file system tool
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            allow_write: false,
            max_file_size: 10 * 1024 * 1024, // 10MB default
        }
    }

    /// Allow write operations
    pub fn with_write_access(mut self) -> Self {
        self.allow_write = true;
        self
    }

    /// Set maximum file size
    pub fn with_max_file_size(mut self, size: usize) -> Self {
        self.max_file_size = size;
        self
    }

    /// Resolve and validate a path
    fn resolve_path(&self, path: &str) -> Result<PathBuf, String> {
        let path = Path::new(path);
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        };

        // Ensure the path is within the base directory (prevent directory traversal)
        let canonical_base = self
            .base_dir
            .canonicalize()
            .map_err(|e| format!("Failed to canonicalize base directory: {}", e))?;

        let canonical_path = full_path
            .canonicalize()
            .or_else(|_| {
                // If file doesn't exist, check parent directory
                if let Some(parent) = full_path.parent() {
                    parent
                        .canonicalize()
                        .map(|p| p.join(full_path.file_name().unwrap_or_default()))
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "Invalid path",
                    ))
                }
            })
            .map_err(|e| format!("Failed to resolve path: {}", e))?;

        if !canonical_path.starts_with(&canonical_base) {
            return Err("Path is outside allowed directory".to_string());
        }

        Ok(canonical_path)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum FileOperation {
    Read { path: String },
    Write { path: String, content: String },
    List { path: Option<String> },
    Delete { path: String },
    Exists { path: String },
    CreateDir { path: String },
}

#[async_trait]
impl ToolExecutor for FileSystemTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let input: Value = serde_json::from_str(arguments)?;
        let operation: FileOperation = serde_json::from_value(input).map_err(|e| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid input: {}", e),
            )) as Box<dyn std::error::Error>
        })?;

        match operation {
            FileOperation::Read { path } => {
                let full_path = self.resolve_path(&path).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
                        as Box<dyn std::error::Error>
                })?;

                // Check file size
                let metadata = fs::metadata(&full_path)
                    .map_err(|e| format!("Failed to get file metadata: {}", e))?;

                if metadata.len() > self.max_file_size as u64 {
                    return Ok(ToolResult::Error(format!(
                        "File too large: {} bytes (max: {} bytes)",
                        metadata.len(),
                        self.max_file_size
                    )));
                }

                let content = fs::read_to_string(&full_path)
                    .map_err(|e| format!("Failed to read file: {}", e))?;

                Ok(ToolResult::Success(serde_json::json!({
                    "path": full_path.display().to_string(),
                    "content": content,
                    "size": metadata.len()
                })))
            }

            FileOperation::Write { path, content } => {
                if !self.allow_write {
                    return Ok(ToolResult::Error(
                        "Write operations are not allowed".to_string(),
                    ));
                }

                let full_path = self.resolve_path(&path).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
                        as Box<dyn std::error::Error>
                })?;

                // Ensure parent directory exists
                if let Some(parent) = full_path.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("Failed to create parent directory: {}", e))?;
                }

                fs::write(&full_path, content)
                    .map_err(|e| format!("Failed to write file: {}", e))?;

                Ok(ToolResult::Success(serde_json::json!({
                    "path": full_path.display().to_string(),
                    "success": true
                })))
            }

            FileOperation::List { path } => {
                let dir_path = if let Some(p) = path {
                    self.resolve_path(&p).map_err(|e| {
                        Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
                            as Box<dyn std::error::Error>
                    })?
                } else {
                    self.base_dir.clone()
                };

                let entries = fs::read_dir(&dir_path)
                    .map_err(|e| format!("Failed to read directory: {}", e))?;

                let mut files = Vec::new();
                let mut dirs = Vec::new();

                for entry in entries {
                    let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
                    let metadata = entry
                        .metadata()
                        .map_err(|e| format!("Failed to get metadata: {}", e))?;

                    let name = entry.file_name().to_string_lossy().to_string();

                    if metadata.is_dir() {
                        dirs.push(name);
                    } else {
                        files.push(serde_json::json!({
                            "name": name,
                            "size": metadata.len()
                        }));
                    }
                }

                Ok(ToolResult::Success(serde_json::json!({
                    "path": dir_path.display().to_string(),
                    "files": files,
                    "directories": dirs
                })))
            }

            FileOperation::Delete { path } => {
                if !self.allow_write {
                    return Ok(ToolResult::Error(
                        "Delete operations are not allowed".to_string(),
                    ));
                }

                let full_path = self.resolve_path(&path).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
                        as Box<dyn std::error::Error>
                })?;

                if full_path.is_dir() {
                    fs::remove_dir_all(&full_path)
                        .map_err(|e| format!("Failed to delete directory: {}", e))?;
                } else {
                    fs::remove_file(&full_path)
                        .map_err(|e| format!("Failed to delete file: {}", e))?;
                }

                Ok(ToolResult::Success(serde_json::json!({
                    "path": full_path.display().to_string(),
                    "deleted": true
                })))
            }

            FileOperation::Exists { path } => {
                let full_path = match self.resolve_path(&path) {
                    Ok(p) => p,
                    Err(_) => {
                        return Ok(ToolResult::Success(serde_json::json!({ "exists": false })))
                    }
                };

                Ok(ToolResult::Success(serde_json::json!({
                    "path": full_path.display().to_string(),
                    "exists": full_path.exists()
                })))
            }

            FileOperation::CreateDir { path } => {
                if !self.allow_write {
                    return Ok(ToolResult::Error(
                        "Create directory operations are not allowed".to_string(),
                    ));
                }

                let full_path = self.resolve_path(&path).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
                        as Box<dyn std::error::Error>
                })?;

                fs::create_dir_all(&full_path)
                    .map_err(|e| format!("Failed to create directory: {}", e))?;

                Ok(ToolResult::Success(serde_json::json!({
                    "path": full_path.display().to_string(),
                    "created": true
                })))
            }
        }
    }

    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "filesystem".to_string(),
            description: Some(
                "Perform file system operations like read, write, list, delete, etc.".to_string(),
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list", "delete", "exists", "create_dir"],
                        "description": "The file system operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (only for write operation)"
                    }
                },
                "required": ["operation", "path"]
            }),
        }
    }
}
