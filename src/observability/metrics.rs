use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Comprehensive metrics for agent operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub agent_id: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens: TokenUsage,
    pub total_cost: f64,
    pub average_response_time: Duration,
    pub tool_usage: HashMap<String, ToolMetrics>,
    pub provider_metrics: HashMap<String, ProviderMetrics>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

impl TokenUsage {
    pub fn new() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }
    }

    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMetrics {
    pub provider_name: String,
    pub model_name: String,
    pub requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub tokens: TokenUsage,
    pub cost: f64,
    pub total_duration: Duration,
    pub average_latency: Duration,
    pub rate_limit_hits: u64,
    pub last_request: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetrics {
    pub tool_name: String,
    pub executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub error_types: HashMap<String, u64>,
}

/// Thread-safe metrics collector
pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, AgentMetrics>>>,
    global_metrics: Arc<RwLock<GlobalMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    pub total_agents: u64,
    pub total_requests: u64,
    pub total_tokens: TokenUsage,
    pub total_cost: f64,
    pub uptime: Duration,
    pub start_time: DateTime<Utc>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            global_metrics: Arc::new(RwLock::new(GlobalMetrics {
                total_agents: 0,
                total_requests: 0,
                total_tokens: TokenUsage::new(),
                total_cost: 0.0,
                uptime: Duration::new(0, 0),
                start_time: Utc::now(),
            })),
        }
    }

    pub fn create_agent_metrics(&self, agent_id: String) {
        let mut metrics = self.metrics.write().unwrap();
        if !metrics.contains_key(&agent_id) {
            let agent_metrics = AgentMetrics {
                agent_id: agent_id.clone(),
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                total_tokens: TokenUsage::new(),
                total_cost: 0.0,
                average_response_time: Duration::new(0, 0),
                tool_usage: HashMap::new(),
                provider_metrics: HashMap::new(),
                created_at: Utc::now(),
                last_updated: Utc::now(),
            };
            metrics.insert(agent_id, agent_metrics);

            // Update global metrics
            let mut global = self.global_metrics.write().unwrap();
            global.total_agents += 1;
        }
    }

    pub fn record_request(
        &self,
        agent_id: &str,
        success: bool,
        duration: Duration,
        tokens: TokenUsage,
        cost: f64,
        provider: &str,
        model: &str,
    ) {
        let mut metrics = self.metrics.write().unwrap();
        if let Some(agent_metrics) = metrics.get_mut(agent_id) {
            agent_metrics.total_requests += 1;
            if success {
                agent_metrics.successful_requests += 1;
            } else {
                agent_metrics.failed_requests += 1;
            }

            agent_metrics.total_tokens.add(&tokens);
            agent_metrics.total_cost += cost;

            // Update average response time
            let total_duration = agent_metrics.average_response_time
                * agent_metrics.total_requests as u32
                + duration;
            agent_metrics.average_response_time =
                total_duration / agent_metrics.total_requests as u32;

            // Update provider metrics
            let provider_key = format!("{}:{}", provider, model);
            let provider_metrics = agent_metrics
                .provider_metrics
                .entry(provider_key)
                .or_insert_with(|| ProviderMetrics {
                    provider_name: provider.to_string(),
                    model_name: model.to_string(),
                    requests: 0,
                    successful_requests: 0,
                    failed_requests: 0,
                    tokens: TokenUsage::new(),
                    cost: 0.0,
                    total_duration: Duration::new(0, 0),
                    average_latency: Duration::new(0, 0),
                    rate_limit_hits: 0,
                    last_request: None,
                });

            provider_metrics.requests += 1;
            if success {
                provider_metrics.successful_requests += 1;
            } else {
                provider_metrics.failed_requests += 1;
            }
            provider_metrics.tokens.add(&tokens);
            provider_metrics.cost += cost;
            provider_metrics.total_duration += duration;
            provider_metrics.average_latency =
                provider_metrics.total_duration / provider_metrics.requests as u32;
            provider_metrics.last_request = Some(Utc::now());

            agent_metrics.last_updated = Utc::now();
        }

        // Update global metrics
        let mut global = self.global_metrics.write().unwrap();
        global.total_requests += 1;
        global.total_tokens.add(&tokens);
        global.total_cost += cost;
        global.uptime = Utc::now()
            .signed_duration_since(global.start_time)
            .to_std()
            .unwrap_or_default();
    }

    pub fn record_tool_execution(
        &self,
        agent_id: &str,
        tool_name: &str,
        success: bool,
        duration: Duration,
        error_type: Option<String>,
    ) {
        let mut metrics = self.metrics.write().unwrap();
        if let Some(agent_metrics) = metrics.get_mut(agent_id) {
            let tool_metrics = agent_metrics
                .tool_usage
                .entry(tool_name.to_string())
                .or_insert_with(|| ToolMetrics {
                    tool_name: tool_name.to_string(),
                    executions: 0,
                    successful_executions: 0,
                    failed_executions: 0,
                    total_duration: Duration::new(0, 0),
                    average_duration: Duration::new(0, 0),
                    error_types: HashMap::new(),
                });

            tool_metrics.executions += 1;
            if success {
                tool_metrics.successful_executions += 1;
            } else {
                tool_metrics.failed_executions += 1;
                if let Some(error) = error_type {
                    *tool_metrics.error_types.entry(error).or_insert(0) += 1;
                }
            }

            tool_metrics.total_duration += duration;
            tool_metrics.average_duration =
                tool_metrics.total_duration / tool_metrics.executions as u32;

            agent_metrics.last_updated = Utc::now();
        }
    }

    pub fn record_rate_limit(&self, agent_id: &str, provider: &str, model: &str) {
        let mut metrics = self.metrics.write().unwrap();
        if let Some(agent_metrics) = metrics.get_mut(agent_id) {
            let provider_key = format!("{}:{}", provider, model);
            if let Some(provider_metrics) = agent_metrics.provider_metrics.get_mut(&provider_key) {
                provider_metrics.rate_limit_hits += 1;
            }
        }
    }

    pub fn get_agent_metrics(&self, agent_id: &str) -> Option<AgentMetrics> {
        self.metrics.read().unwrap().get(agent_id).cloned()
    }

    pub fn get_all_agent_metrics(&self) -> HashMap<String, AgentMetrics> {
        self.metrics.read().unwrap().clone()
    }

    pub fn get_global_metrics(&self) -> GlobalMetrics {
        self.global_metrics.read().unwrap().clone()
    }

    pub fn reset_agent_metrics(&self, agent_id: &str) {
        let mut metrics = self.metrics.write().unwrap();
        if let Some(agent_metrics) = metrics.get_mut(agent_id) {
            agent_metrics.total_requests = 0;
            agent_metrics.successful_requests = 0;
            agent_metrics.failed_requests = 0;
            agent_metrics.total_tokens = TokenUsage::new();
            agent_metrics.total_cost = 0.0;
            agent_metrics.average_response_time = Duration::new(0, 0);
            agent_metrics.tool_usage.clear();
            agent_metrics.provider_metrics.clear();
            agent_metrics.last_updated = Utc::now();
        }
    }

    pub fn export_metrics(&self) -> serde_json::Value {
        let agent_metrics = self.get_all_agent_metrics();
        let global_metrics = self.get_global_metrics();

        serde_json::json!({
            "global": global_metrics,
            "agents": agent_metrics,
            "exported_at": Utc::now()
        })
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage() {
        let mut usage = TokenUsage::new();
        assert_eq!(usage.total(), 0);

        usage.input_tokens = 100;
        usage.output_tokens = 50;
        assert_eq!(usage.total(), 150);

        let other = TokenUsage {
            input_tokens: 200,
            output_tokens: 100,
            cache_read_tokens: 50,
            cache_write_tokens: 25,
        };

        usage.add(&other);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 150);
        assert_eq!(usage.cache_read_tokens, 50);
        assert_eq!(usage.cache_write_tokens, 25);
        assert_eq!(usage.total(), 525);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        let agent_id = "test-agent";

        collector.create_agent_metrics(agent_id.to_string());

        let tokens = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        };

        collector.record_request(
            agent_id,
            true,
            Duration::from_millis(500),
            tokens,
            0.01,
            "openai",
            "gpt-4",
        );

        let metrics = collector.get_agent_metrics(agent_id).unwrap();
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.total_tokens.input_tokens, 100);
        assert_eq!(metrics.total_cost, 0.01);

        let global = collector.get_global_metrics();
        assert_eq!(global.total_agents, 1);
        assert_eq!(global.total_requests, 1);
    }
}
