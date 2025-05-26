use crate::{Message, Role, MessageContent};

/// A message in the context with additional metadata
#[derive(Clone, Debug)]
pub struct ContextMessage {
    pub message: Message,
    pub timestamp: std::time::SystemTime,
    pub metadata: Option<serde_json::Value>,
}

/// Manages the conversation context for an agent
#[derive(Clone, Debug)]
pub struct Context {
    messages: Vec<ContextMessage>,
    max_messages: Option<usize>,
    max_tokens: Option<usize>,
}

impl Context {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: None,
            max_tokens: None,
        }
    }

    /// Create a context with limits
    pub fn with_limits(max_messages: Option<usize>, max_tokens: Option<usize>) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
            max_tokens,
        }
    }

    /// Add a system message
    pub fn add_system_message(&mut self, content: &str) {
        self.add_message(Message {
            role: Role::System,
            content: MessageContent::text(content),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: &str) {
        self.add_message(Message {
            role: Role::User,
            content: MessageContent::text(content),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Add an assistant message
    pub fn add_assistant_message(&mut self, content: &str) {
        self.add_message(Message {
            role: Role::Assistant,
            content: MessageContent::text(content),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Add a tool result message
    pub fn add_tool_result(&mut self, tool_call_id: &str, result: &str) {
        self.add_message(Message {
            role: Role::Tool,
            content: MessageContent::text(result),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        });
    }

    /// Add a memory context (as a system message)
    pub fn add_memory(&mut self, memory: String) {
        self.add_message(Message {
            role: Role::System,
            content: MessageContent::text(format!("[Memory] {}", memory)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Add a message with metadata
    pub fn add_message_with_metadata(
        &mut self, 
        message: Message, 
        metadata: Option<serde_json::Value>
    ) {
        let context_msg = ContextMessage {
            message,
            timestamp: std::time::SystemTime::now(),
            metadata,
        };
        
        self.messages.push(context_msg);
        self.enforce_limits();
    }

    /// Add a message
    pub fn add_message(&mut self, message: Message) {
        self.add_message_with_metadata(message, None);
    }

    /// Get all messages
    pub fn messages(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter().map(|cm| &cm.message)
    }

    /// Get mutable access to messages
    pub fn messages_mut(&mut self) -> &mut Vec<ContextMessage> {
        &mut self.messages
    }

    /// Convert to a vector of messages for API calls
    pub fn to_messages(&self) -> Vec<Message> {
        self.messages.iter().map(|cm| cm.message.clone()).collect()
    }

    /// Clear all messages except system messages
    pub fn clear(&mut self) {
        self.messages.retain(|cm| matches!(cm.message.role, Role::System));
    }

    /// Clear all messages
    pub fn clear_all(&mut self) {
        self.messages.clear();
    }

    /// Get the number of messages
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if the context is empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Estimate token count (rough approximation)
    pub fn estimate_tokens(&self) -> usize {
        self.messages.iter()
            .map(|cm| {
                let content_len = match &cm.message.content {
                    MessageContent::Text(text) => text.len() / 4, // Rough estimate
                    MessageContent::Parts(parts) => {
                        parts.iter().map(|p| match p {
                            crate::ContentPart::Text { text } => text.len() / 4,
                            crate::ContentPart::Image { .. } => 100, // Rough estimate for image
                        }).sum()
                    }
                };
                
                // Add some overhead for role and structure
                content_len + 10
            })
            .sum()
    }

    /// Enforce message and token limits
    fn enforce_limits(&mut self) {
        // Keep system messages at the beginning
        let system_count = self.messages.iter()
            .filter(|cm| matches!(cm.message.role, Role::System))
            .count();
        
        // Enforce message limit
        if let Some(max) = self.max_messages {
            if self.messages.len() > max {
                // Remove oldest non-system messages
                let to_remove = self.messages.len() - max;
                let mut removed = 0;
                
                self.messages.retain(|cm| {
                    if removed >= to_remove || matches!(cm.message.role, Role::System) {
                        true
                    } else {
                        removed += 1;
                        false
                    }
                });
            }
        }
        
        // Enforce token limit (rough)
        if let Some(max_tokens) = self.max_tokens {
            while self.estimate_tokens() > max_tokens && self.messages.len() > system_count {
                // Find first non-system message and remove it
                if let Some(pos) = self.messages.iter().position(|cm| {
                    !matches!(cm.message.role, Role::System)
                }) {
                    self.messages.remove(pos);
                } else {
                    break;
                }
            }
        }
    }

    /// Create a summary of the context
    pub fn summary(&self) -> String {
        format!(
            "Context: {} messages, ~{} tokens",
            self.messages.len(),
            self.estimate_tokens()
        )
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_limits() {
        let mut ctx = Context::with_limits(Some(3), None);
        
        ctx.add_system_message("System prompt");
        ctx.add_user_message("Message 1");
        ctx.add_assistant_message("Response 1");
        ctx.add_user_message("Message 2");
        
        assert_eq!(ctx.len(), 3); // Should have removed oldest non-system message
        assert_eq!(ctx.messages().next().unwrap().role, Role::System);
    }
    
    #[test]
    fn test_context_clear() {
        let mut ctx = Context::new();
        
        ctx.add_system_message("System prompt");
        ctx.add_user_message("User message");
        ctx.add_assistant_message("Assistant response");
        
        ctx.clear();
        
        assert_eq!(ctx.len(), 1); // Only system message remains
        assert_eq!(ctx.messages().next().unwrap().role, Role::System);
    }
}