use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{AnyPool, Row, Column};
use std::collections::HashMap;

use crate::agent::tools::{ToolExecutor, ToolResult};
use crate::ToolFunction;

/// Database query tool for executing SQL queries
pub struct DatabaseTool {
    pool: AnyPool,
    /// Whether to allow write operations (INSERT, UPDATE, DELETE)
    allow_write: bool,
    /// Maximum number of rows to return
    max_rows: usize,
    /// Query timeout in seconds
    timeout_secs: u64,
}

impl DatabaseTool {
    /// Create a new database tool
    pub fn new(pool: AnyPool) -> Self {
        Self {
            pool,
            allow_write: false,
            max_rows: 1000,
            timeout_secs: 30,
        }
    }
    
    /// Allow write operations
    pub fn with_write_access(mut self) -> Self {
        self.allow_write = true;
        self
    }
    
    /// Set maximum rows to return
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.max_rows = max_rows;
        self
    }
    
    /// Set query timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
    
    /// Check if query is read-only
    fn is_read_only(sql: &str) -> bool {
        let sql_upper = sql.trim().to_uppercase();
        sql_upper.starts_with("SELECT") || 
        sql_upper.starts_with("WITH") ||
        sql_upper.starts_with("SHOW") ||
        sql_upper.starts_with("DESCRIBE") ||
        sql_upper.starts_with("EXPLAIN")
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum DatabaseOperation {
    Query {
        sql: String,
        params: Option<Vec<Value>>,
    },
    Schema {
        table: Option<String>,
    },
    Tables,
}

#[async_trait]
impl ToolExecutor for DatabaseTool {
    async fn execute(&self, arguments: &str) -> Result<ToolResult, Box<dyn std::error::Error>> {
        let input: Value = serde_json::from_str(arguments)?;
        let operation: DatabaseOperation = serde_json::from_value(input)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid input: {}", e))) as Box<dyn std::error::Error>)?;
        
        match operation {
            DatabaseOperation::Query { sql, params } => {
                // Check permissions
                if !self.allow_write && !Self::is_read_only(&sql) {
                    return Ok(ToolResult::Error("Write operations are not allowed".to_string()));
                }
                
                // Build query
                let mut query = sqlx::query(&sql);
                
                // Bind parameters if provided
                if let Some(params) = params {
                    for param in params {
                        query = match param {
                            Value::Null => query.bind(None::<String>),
                            Value::Bool(b) => query.bind(b),
                            Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    query.bind(i)
                                } else if let Some(f) = n.as_f64() {
                                    query.bind(f)
                                } else {
                                    return Ok(ToolResult::Error("Invalid number parameter".to_string()));
                                }
                            }
                            Value::String(s) => query.bind(s),
                            _ => return Ok(ToolResult::Error("Invalid parameter type".to_string())),
                        };
                    }
                }
                
                // Execute query
                let rows = query
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Query failed: {}", e))) as Box<dyn std::error::Error>)?;
                
                // Check row limit
                if rows.len() > self.max_rows {
                    return Ok(ToolResult::Error(format!(
                        "Query returned too many rows: {} (max: {})",
                        rows.len(),
                        self.max_rows
                    )));
                }
                
                // Convert rows to JSON
                let mut results = Vec::new();
                for row in rows.iter() {
                    let mut row_map = HashMap::new();
                    
                    for (i, column) in row.columns().iter().enumerate() {
                        let value: Value = if let Ok(v) = row.try_get::<String, _>(i) {
                            Value::String(v)
                        } else if let Ok(v) = row.try_get::<i64, _>(i) {
                            Value::Number(v.into())
                        } else if let Ok(v) = row.try_get::<f64, _>(i) {
                            serde_json::Number::from_f64(v)
                                .map(Value::Number)
                                .unwrap_or(Value::Null)
                        } else if let Ok(v) = row.try_get::<bool, _>(i) {
                            Value::Bool(v)
                        } else if let Ok(v) = row.try_get::<Option<String>, _>(i) {
                            v.map(Value::String).unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        };
                        
                        row_map.insert(column.name().to_string(), value);
                    }
                    
                    results.push(Value::Object(row_map.into_iter().collect()));
                }
                
                Ok(ToolResult::Success(serde_json::json!({
                    "rows": results,
                    "row_count": results.len()
                })))
            }
            
            DatabaseOperation::Schema { table } => {
                let schema_query = if let Some(ref table_name) = table {
                    // Get schema for specific table
                    // Use dynamic query based on database type detection
                    // For now, we'll use a generic approach
                    {
                            format!(
                                "SELECT column_name, data_type, is_nullable 
                                 FROM information_schema.columns 
                                 WHERE table_name = '{}'
                                 ORDER BY ordinal_position",
                                table_name
                            )
                    }
                } else {
                    return Ok(ToolResult::Error("Table name required for schema query".to_string()));
                };
                
                let rows = sqlx::query(&schema_query)
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Schema query failed: {}", e))) as Box<dyn std::error::Error>)?;
                
                let mut columns = Vec::new();
                for row in rows.iter() {
                    // Generic column info extraction
                    let column_info = serde_json::json!({
                        "name": row.try_get::<String, _>(0).unwrap_or_default(),
                        "type": row.try_get::<String, _>(1).unwrap_or_default(),
                        "nullable": row.try_get::<String, _>(2).unwrap_or_default() == "YES"
                    });
                    columns.push(column_info);
                }
                
                Ok(ToolResult::Success(serde_json::json!({
                    "table": table,
                    "columns": columns
                })))
            }
            
            DatabaseOperation::Tables => {
                // Use generic table query - this will work for PostgreSQL and MySQL
                let tables_query = "SELECT table_name FROM information_schema.tables 
                                   WHERE table_schema = 'public' OR table_schema = DATABASE() 
                                   ORDER BY table_name";
                
                let rows = sqlx::query(tables_query)
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Tables query failed: {}", e))) as Box<dyn std::error::Error>)?;
                
                let mut tables = Vec::new();
                for row in rows.iter() {
                    if let Ok(table_name) = row.try_get::<String, _>(0) {
                        tables.push(table_name);
                    }
                }
                
                Ok(ToolResult::Success(serde_json::json!({
                    "tables": tables
                })))
            }
        }
    }
    
    fn definition(&self) -> ToolFunction {
        ToolFunction {
            name: "database".to_string(),
            description: Some("Execute SQL queries and explore database schema".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["query", "schema", "tables"],
                        "description": "The database operation to perform"
                    },
                    "sql": {
                        "type": "string",
                        "description": "SQL query to execute (only for query operation)"
                    },
                    "params": {
                        "type": "array",
                        "description": "Optional query parameters (only for query operation)",
                        "items": {}
                    },
                    "table": {
                        "type": "string",
                        "description": "Table name to get schema for (only for schema operation)"
                    }
                },
                "required": ["operation"]
            }),
        }
    }
}

