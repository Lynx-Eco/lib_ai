use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Represents a single trace event in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration: Option<Duration>,
    pub status: TraceStatus,
    pub tags: HashMap<String, String>,
    pub logs: Vec<TraceLog>,
    pub baggage: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceStatus {
    Ok,
    Error,
    Cancelled,
    DeadlineExceeded,
    InvalidArgument,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceLog {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Active span that tracks timing and can be finished
pub struct TraceSpan {
    pub event: TraceEvent,
    start_instant: Instant,
    tracer: Arc<AgentTracer>,
}

impl TraceSpan {
    pub fn set_tag(&mut self, key: String, value: String) {
        self.event.tags.insert(key, value);
    }
    
    pub fn set_baggage(&mut self, key: String, value: String) {
        self.event.baggage.insert(key, value);
    }
    
    pub fn log(&mut self, level: LogLevel, message: String, fields: HashMap<String, String>) {
        let log = TraceLog {
            timestamp: Utc::now(),
            level,
            message,
            fields,
        };
        self.event.logs.push(log);
    }
    
    pub fn log_info(&mut self, message: String) {
        self.log(LogLevel::Info, message, HashMap::new());
    }
    
    pub fn log_error(&mut self, message: String) {
        self.log(LogLevel::Error, message, HashMap::new());
        self.event.status = TraceStatus::Error;
    }
    
    pub fn set_status(&mut self, status: TraceStatus) {
        self.event.status = status;
    }
    
    pub fn finish(mut self) {
        self.event.end_time = Some(Utc::now());
        self.event.duration = Some(self.start_instant.elapsed());
        self.tracer.finish_span(self.event);
    }
    
    pub fn child_span(&self, operation_name: String) -> TraceSpan {
        self.tracer.start_span_with_parent(operation_name, Some(self.event.span_id.clone()))
    }
}

/// Agent tracer for collecting distributed traces
pub struct AgentTracer {
    traces: Arc<RwLock<HashMap<String, Vec<TraceEvent>>>>,
    current_trace: Arc<RwLock<Option<String>>>,
    config: TracingConfig,
}

#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub enabled: bool,
    pub sample_rate: f64, // 0.0 to 1.0
    pub max_traces: usize,
    pub max_spans_per_trace: usize,
    pub export_interval: Duration,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 1.0,
            max_traces: 1000,
            max_spans_per_trace: 100,
            export_interval: Duration::from_secs(60),
        }
    }
}

impl AgentTracer {
    pub fn new(config: TracingConfig) -> Self {
        Self {
            traces: Arc::new(RwLock::new(HashMap::new())),
            current_trace: Arc::new(RwLock::new(None)),
            config,
        }
    }
    
    pub fn start_trace(&self, operation_name: String) -> Option<TraceSpan> {
        if !self.config.enabled || !self.should_sample() {
            return None;
        }
        
        let trace_id = Uuid::new_v4().to_string();
        *self.current_trace.write().unwrap() = Some(trace_id.clone());
        
        Some(self.start_span_with_trace(operation_name, trace_id, None))
    }
    
    pub fn start_span(&self, operation_name: String) -> Option<TraceSpan> {
        if !self.config.enabled {
            return None;
        }
        
        let current_trace = self.current_trace.read().unwrap().clone();
        if let Some(trace_id) = current_trace {
            Some(self.start_span_with_trace(operation_name, trace_id, None))
        } else {
            self.start_trace(operation_name)
        }
    }
    
    pub fn start_span_with_parent(&self, operation_name: String, parent_span_id: Option<String>) -> TraceSpan {
        let current_trace = self.current_trace.read().unwrap().clone();
        let trace_id = current_trace.unwrap_or_else(|| Uuid::new_v4().to_string());
        
        self.start_span_with_trace(operation_name, trace_id, parent_span_id)
    }
    
    fn start_span_with_trace(&self, operation_name: String, trace_id: String, parent_span_id: Option<String>) -> TraceSpan {
        let span_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        let event = TraceEvent {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id,
            operation_name,
            start_time: now,
            end_time: None,
            duration: None,
            status: TraceStatus::Ok,
            tags: HashMap::new(),
            logs: Vec::new(),
            baggage: HashMap::new(),
        };
        
        TraceSpan {
            event,
            start_instant: Instant::now(),
            tracer: Arc::new(self.clone()),
        }
    }
    
    fn finish_span(&self, event: TraceEvent) {
        let mut traces = self.traces.write().unwrap();
        let trace_spans = traces.entry(event.trace_id.clone()).or_insert_with(Vec::new);
        
        // Limit spans per trace
        if trace_spans.len() < self.config.max_spans_per_trace {
            trace_spans.push(event);
        }
        
        // Limit total traces
        if traces.len() > self.config.max_traces {
            // Remove oldest trace (simple FIFO)
            if let Some(oldest_trace_id) = traces.keys().next().cloned() {
                traces.remove(&oldest_trace_id);
            }
        }
    }
    
    fn should_sample(&self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen::<f64>() < self.config.sample_rate
    }
    
    pub fn get_trace(&self, trace_id: &str) -> Option<Vec<TraceEvent>> {
        self.traces.read().unwrap().get(trace_id).cloned()
    }
    
    pub fn get_all_traces(&self) -> HashMap<String, Vec<TraceEvent>> {
        self.traces.read().unwrap().clone()
    }
    
    pub fn clear_traces(&self) {
        self.traces.write().unwrap().clear();
    }
    
    pub fn export_traces(&self) -> serde_json::Value {
        let traces = self.get_all_traces();
        serde_json::json!({
            "traces": traces,
            "exported_at": Utc::now(),
            "config": {
                "sample_rate": self.config.sample_rate,
                "max_traces": self.config.max_traces,
                "max_spans_per_trace": self.config.max_spans_per_trace
            }
        })
    }
    
    pub fn finish_trace(&self) {
        *self.current_trace.write().unwrap() = None;
    }
}

impl Clone for AgentTracer {
    fn clone(&self) -> Self {
        Self {
            traces: self.traces.clone(),
            current_trace: self.current_trace.clone(),
            config: self.config.clone(),
        }
    }
}

/// Convenience macros for tracing
#[macro_export]
macro_rules! trace_span {
    ($tracer:expr, $name:expr) => {
        $tracer.start_span($name.to_string())
    };
    ($tracer:expr, $name:expr, $($key:expr => $value:expr),*) => {{
        let mut span = $tracer.start_span($name.to_string())?;
        $(
            span.set_tag($key.to_string(), $value.to_string());
        )*
        Some(span)
    }};
}

#[macro_export]
macro_rules! trace_log {
    ($span:expr, $level:expr, $msg:expr) => {
        $span.log($level, $msg.to_string(), std::collections::HashMap::new());
    };
    ($span:expr, $level:expr, $msg:expr, $($key:expr => $value:expr),*) => {{
        let mut fields = std::collections::HashMap::new();
        $(
            fields.insert($key.to_string(), $value.to_string());
        )*
        $span.log($level, $msg.to_string(), fields);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_trace_creation() {
        let config = TracingConfig::default();
        let tracer = AgentTracer::new(config);
        
        let span = tracer.start_trace("test_operation".to_string()).unwrap();
        let trace_id = span.event.trace_id.clone();
        span.finish();
        
        let traces = tracer.get_trace(&trace_id);
        assert!(traces.is_some());
        assert_eq!(traces.unwrap().len(), 1);
    }
    
    #[test]
    fn test_nested_spans() {
        let config = TracingConfig::default();
        let tracer = AgentTracer::new(config);
        
        let mut parent_span = tracer.start_trace("parent_operation".to_string()).unwrap();
        parent_span.set_tag("operation".to_string(), "parent".to_string());
        
        let mut child_span = parent_span.child_span("child_operation".to_string());
        child_span.set_tag("operation".to_string(), "child".to_string());
        child_span.log_info("Child operation started".to_string());
        
        thread::sleep(Duration::from_millis(10));
        
        child_span.finish();
        parent_span.finish();
        
        let trace_id = parent_span.event.trace_id;
        let traces = tracer.get_trace(&trace_id).unwrap();
        assert_eq!(traces.len(), 2);
        
        // Check parent-child relationship
        let child = traces.iter().find(|t| t.operation_name == "child_operation").unwrap();
        assert!(child.parent_span_id.is_some());
    }
    
    #[test]
    fn test_sampling() {
        let config = TracingConfig {
            sample_rate: 0.0,
            ..Default::default()
        };
        let tracer = AgentTracer::new(config);
        
        let span = tracer.start_trace("test_operation".to_string());
        assert!(span.is_none());
    }
}