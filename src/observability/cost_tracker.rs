use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cost tracking for different AI providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTracker {
    pub provider_costs: HashMap<String, ProviderCosts>,
    pub total_cost: f64,
    pub start_time: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCosts {
    pub provider_name: String,
    pub models: HashMap<String, ModelCosts>,
    pub total_cost: f64,
    pub total_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCosts {
    pub model_name: String,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_read_cost: f64,
    pub cache_write_cost: f64,
    pub total_cost: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub requests: u64,
}

/// Pricing information for different providers and models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingInfo {
    pub provider: String,
    pub model: String,
    pub input_price_per_1k_tokens: f64,
    pub output_price_per_1k_tokens: f64,
    pub cache_read_price_per_1k_tokens: Option<f64>,
    pub cache_write_price_per_1k_tokens: Option<f64>,
    pub currency: String,
    pub last_updated: DateTime<Utc>,
}

impl PricingInfo {
    pub fn calculate_cost(
        &self,
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
    ) -> f64 {
        let input_cost = (input_tokens as f64 / 1000.0) * self.input_price_per_1k_tokens;
        let output_cost = (output_tokens as f64 / 1000.0) * self.output_price_per_1k_tokens;
        let cache_read_cost = (cache_read_tokens as f64 / 1000.0)
            * self.cache_read_price_per_1k_tokens.unwrap_or(0.0);
        let cache_write_cost = (cache_write_tokens as f64 / 1000.0)
            * self.cache_write_price_per_1k_tokens.unwrap_or(0.0);

        input_cost + output_cost + cache_read_cost + cache_write_cost
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            provider_costs: HashMap::new(),
            total_cost: 0.0,
            start_time: Utc::now(),
            last_updated: Utc::now(),
        }
    }

    pub fn record_usage(
        &mut self,
        provider: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        pricing: &PricingInfo,
    ) {
        let input_cost = (input_tokens as f64 / 1000.0) * pricing.input_price_per_1k_tokens;
        let output_cost = (output_tokens as f64 / 1000.0) * pricing.output_price_per_1k_tokens;
        let cache_read_cost = (cache_read_tokens as f64 / 1000.0)
            * pricing.cache_read_price_per_1k_tokens.unwrap_or(0.0);
        let cache_write_cost = (cache_write_tokens as f64 / 1000.0)
            * pricing.cache_write_price_per_1k_tokens.unwrap_or(0.0);

        let total_request_cost = input_cost + output_cost + cache_read_cost + cache_write_cost;

        // Update provider costs
        let provider_costs = self
            .provider_costs
            .entry(provider.to_string())
            .or_insert_with(|| ProviderCosts {
                provider_name: provider.to_string(),
                models: HashMap::new(),
                total_cost: 0.0,
                total_requests: 0,
            });

        provider_costs.total_cost += total_request_cost;
        provider_costs.total_requests += 1;

        // Update model costs
        let model_costs = provider_costs
            .models
            .entry(model.to_string())
            .or_insert_with(|| ModelCosts {
                model_name: model.to_string(),
                input_cost: 0.0,
                output_cost: 0.0,
                cache_read_cost: 0.0,
                cache_write_cost: 0.0,
                total_cost: 0.0,
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                requests: 0,
            });

        model_costs.input_cost += input_cost;
        model_costs.output_cost += output_cost;
        model_costs.cache_read_cost += cache_read_cost;
        model_costs.cache_write_cost += cache_write_cost;
        model_costs.total_cost += total_request_cost;
        model_costs.input_tokens += input_tokens;
        model_costs.output_tokens += output_tokens;
        model_costs.cache_read_tokens += cache_read_tokens;
        model_costs.cache_write_tokens += cache_write_tokens;
        model_costs.requests += 1;

        // Update total cost
        self.total_cost += total_request_cost;
        self.last_updated = Utc::now();
    }

    pub fn get_cost_by_provider(&self, provider: &str) -> Option<&ProviderCosts> {
        self.provider_costs.get(provider)
    }

    pub fn get_cost_by_model(&self, provider: &str, model: &str) -> Option<&ModelCosts> {
        self.provider_costs.get(provider)?.models.get(model)
    }

    pub fn get_pricing(&self, provider: &str, model: &str) -> PricingInfo {
        let default_pricing = get_default_pricing();
        let key = format!("{}:{}", provider, model);

        default_pricing.get(&key).cloned().unwrap_or_else(|| {
            // Fallback pricing for unknown models
            PricingInfo {
                provider: provider.to_string(),
                model: model.to_string(),
                input_price_per_1k_tokens: 0.001, // $1 per 1M tokens
                output_price_per_1k_tokens: 0.002, // $2 per 1M tokens
                cache_read_price_per_1k_tokens: None,
                cache_write_price_per_1k_tokens: None,
                currency: "USD".to_string(),
                last_updated: Utc::now(),
            }
        })
    }

    pub fn generate_report(&self) -> CostReport {
        let mut provider_breakdown = Vec::new();

        for (provider_name, provider_costs) in &self.provider_costs {
            let mut model_breakdown = Vec::new();

            for (model_name, model_costs) in &provider_costs.models {
                model_breakdown.push(ModelReportEntry {
                    model_name: model_name.clone(),
                    total_cost: model_costs.total_cost,
                    requests: model_costs.requests,
                    input_tokens: model_costs.input_tokens,
                    output_tokens: model_costs.output_tokens,
                    cache_read_tokens: model_costs.cache_read_tokens,
                    cache_write_tokens: model_costs.cache_write_tokens,
                    cost_per_request: model_costs.total_cost / model_costs.requests.max(1) as f64,
                    cost_per_token: model_costs.total_cost
                        / (model_costs.input_tokens
                            + model_costs.output_tokens
                            + model_costs.cache_read_tokens
                            + model_costs.cache_write_tokens)
                            .max(1) as f64,
                });
            }

            provider_breakdown.push(ProviderReportEntry {
                provider_name: provider_name.clone(),
                total_cost: provider_costs.total_cost,
                requests: provider_costs.total_requests,
                models: model_breakdown,
                cost_percentage: (provider_costs.total_cost / self.total_cost.max(0.001)) * 100.0,
            });
        }

        // Sort by cost descending
        provider_breakdown.sort_by(|a, b| b.total_cost.partial_cmp(&a.total_cost).unwrap());

        CostReport {
            total_cost: self.total_cost,
            start_time: self.start_time,
            end_time: self.last_updated,
            duration: self
                .last_updated
                .signed_duration_since(self.start_time)
                .to_std()
                .unwrap_or_default(),
            providers: provider_breakdown,
            generated_at: Utc::now(),
        }
    }

    pub fn reset(&mut self) {
        self.provider_costs.clear();
        self.total_cost = 0.0;
        self.start_time = Utc::now();
        self.last_updated = Utc::now();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostReport {
    pub total_cost: f64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration: std::time::Duration,
    pub providers: Vec<ProviderReportEntry>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderReportEntry {
    pub provider_name: String,
    pub total_cost: f64,
    pub requests: u64,
    pub cost_percentage: f64,
    pub models: Vec<ModelReportEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReportEntry {
    pub model_name: String,
    pub total_cost: f64,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_per_request: f64,
    pub cost_per_token: f64,
}

/// Default pricing information for popular providers
pub fn get_default_pricing() -> HashMap<String, PricingInfo> {
    let mut pricing = HashMap::new();

    // OpenAI GPT-4o
    pricing.insert(
        "openai:gpt-4o".to_string(),
        PricingInfo {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            input_price_per_1k_tokens: 0.0025,
            output_price_per_1k_tokens: 0.01,
            cache_read_price_per_1k_tokens: None,
            cache_write_price_per_1k_tokens: None,
            currency: "USD".to_string(),
            last_updated: Utc::now(),
        },
    );

    // OpenAI GPT-4o-mini
    pricing.insert(
        "openai:gpt-4o-mini".to_string(),
        PricingInfo {
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            input_price_per_1k_tokens: 0.00015,
            output_price_per_1k_tokens: 0.0006,
            cache_read_price_per_1k_tokens: None,
            cache_write_price_per_1k_tokens: None,
            currency: "USD".to_string(),
            last_updated: Utc::now(),
        },
    );

    // Anthropic Claude 3.5 Sonnet
    pricing.insert(
        "anthropic:claude-3-5-sonnet-20241022".to_string(),
        PricingInfo {
            provider: "anthropic".to_string(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            input_price_per_1k_tokens: 0.003,
            output_price_per_1k_tokens: 0.015,
            cache_read_price_per_1k_tokens: Some(0.0003),
            cache_write_price_per_1k_tokens: Some(0.00375),
            currency: "USD".to_string(),
            last_updated: Utc::now(),
        },
    );

    // Anthropic Claude 3.5 Haiku
    pricing.insert(
        "anthropic:claude-3-5-haiku-20241022".to_string(),
        PricingInfo {
            provider: "anthropic".to_string(),
            model: "claude-3-5-haiku-20241022".to_string(),
            input_price_per_1k_tokens: 0.001,
            output_price_per_1k_tokens: 0.005,
            cache_read_price_per_1k_tokens: Some(0.0001),
            cache_write_price_per_1k_tokens: Some(0.00125),
            currency: "USD".to_string(),
            last_updated: Utc::now(),
        },
    );

    // Google Gemini 1.5 Pro
    pricing.insert(
        "google:gemini-1.5-pro".to_string(),
        PricingInfo {
            provider: "google".to_string(),
            model: "gemini-1.5-pro".to_string(),
            input_price_per_1k_tokens: 0.001,
            output_price_per_1k_tokens: 0.002,
            cache_read_price_per_1k_tokens: None,
            cache_write_price_per_1k_tokens: None,
            currency: "USD".to_string(),
            last_updated: Utc::now(),
        },
    );

    pricing
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_calculation() {
        let mut tracker = CostTracker::new();
        let pricing = PricingInfo {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            input_price_per_1k_tokens: 0.0025,
            output_price_per_1k_tokens: 0.01,
            cache_read_price_per_1k_tokens: None,
            cache_write_price_per_1k_tokens: None,
            currency: "USD".to_string(),
            last_updated: Utc::now(),
        };

        // Record usage: 1000 input tokens, 500 output tokens
        tracker.record_usage("openai", "gpt-4o", 1000, 500, 0, 0, &pricing);

        // Expected cost: (1000/1000 * 0.0025) + (500/1000 * 0.01) = 0.0025 + 0.005 = 0.0075
        assert!((tracker.total_cost - 0.0075).abs() < 0.0001);

        let model_costs = tracker.get_cost_by_model("openai", "gpt-4o").unwrap();
        assert_eq!(model_costs.input_tokens, 1000);
        assert_eq!(model_costs.output_tokens, 500);
        assert_eq!(model_costs.requests, 1);
    }

    #[test]
    fn test_cost_report() {
        let mut tracker = CostTracker::new();
        let pricing = get_default_pricing();

        // Record some usage
        tracker.record_usage(
            "openai",
            "gpt-4o",
            1000,
            500,
            0,
            0,
            pricing.get("openai:gpt-4o").unwrap(),
        );
        tracker.record_usage(
            "anthropic",
            "claude-3-5-sonnet-20241022",
            800,
            300,
            100,
            50,
            pricing.get("anthropic:claude-3-5-sonnet-20241022").unwrap(),
        );

        let report = tracker.generate_report();
        assert_eq!(report.providers.len(), 2);
        assert!(report.total_cost > 0.0);

        // Check that percentages add up to 100%
        let total_percentage: f64 = report.providers.iter().map(|p| p.cost_percentage).sum();
        assert!((total_percentage - 100.0).abs() < 0.01);
    }
}
