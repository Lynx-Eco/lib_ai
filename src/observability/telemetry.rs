use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::{AgentTracer, CostTracker, MetricsCollector};

/// Configuration for telemetry export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub export_interval: Duration,
    pub exporters: Vec<ExporterConfig>,
    pub batch_size: usize,
    pub max_queue_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExporterConfig {
    pub name: String,
    pub exporter_type: ExporterType,
    pub endpoint: Option<String>,
    pub headers: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExporterType {
    Console,
    File {
        path: String,
    },
    Http {
        endpoint: String,
        format: HttpFormat,
    },
    Jaeger {
        endpoint: String,
    },
    Prometheus {
        endpoint: String,
    },
    OpenTelemetry {
        endpoint: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpFormat {
    Json,
    JsonLines,
    Protobuf,
}

/// Main telemetry system that coordinates all observability components
pub struct TelemetryExporter {
    config: TelemetryConfig,
    metrics_collector: Arc<MetricsCollector>,
    tracer: Arc<AgentTracer>,
    cost_tracker: Arc<RwLock<CostTracker>>,
    exporters: Vec<Box<dyn Exporter>>,
    running: Arc<RwLock<bool>>,
}

impl TelemetryExporter {
    pub fn new(
        config: TelemetryConfig,
        metrics_collector: Arc<MetricsCollector>,
        tracer: Arc<AgentTracer>,
        cost_tracker: Arc<RwLock<CostTracker>>,
    ) -> Self {
        let mut exporters: Vec<Box<dyn Exporter>> = Vec::new();

        for exporter_config in &config.exporters {
            if exporter_config.enabled {
                match &exporter_config.exporter_type {
                    ExporterType::Console => {
                        exporters.push(Box::new(ConsoleExporter::new()));
                    }
                    ExporterType::File { path } => {
                        exporters.push(Box::new(FileExporter::new(path.clone())));
                    }
                    ExporterType::Http { endpoint, format } => {
                        exporters.push(Box::new(HttpExporter::new(
                            endpoint.clone(),
                            format.clone(),
                            exporter_config.headers.clone(),
                        )));
                    }
                    ExporterType::Jaeger { endpoint } => {
                        exporters.push(Box::new(JaegerExporter::new(endpoint.clone())));
                    }
                    ExporterType::Prometheus { endpoint } => {
                        exporters.push(Box::new(PrometheusExporter::new(endpoint.clone())));
                    }
                    ExporterType::OpenTelemetry { endpoint } => {
                        exporters.push(Box::new(OpenTelemetryExporter::new(endpoint.clone())));
                    }
                }
            }
        }

        Self {
            config,
            metrics_collector,
            tracer,
            cost_tracker,
            exporters,
            running: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn start(&self) {
        if !self.config.enabled {
            return;
        }

        *self.running.write().await = true;

        let metrics_collector = self.metrics_collector.clone();
        let tracer = self.tracer.clone();
        let cost_tracker = self.cost_tracker.clone();
        let exporters = self
            .exporters
            .iter()
            .map(|e| e.clone_box())
            .collect::<Vec<_>>();
        let export_interval = self.config.export_interval;
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(export_interval);

            while *running.read().await {
                interval.tick().await;

                // Collect all telemetry data
                let telemetry_data = TelemetryData {
                    timestamp: Utc::now(),
                    metrics: metrics_collector.export_metrics(),
                    traces: tracer.export_traces(),
                    costs: {
                        let tracker = cost_tracker.read().await;
                        serde_json::to_value(tracker.generate_report()).unwrap_or_default()
                    },
                };

                // Export to all configured exporters
                for exporter in &exporters {
                    if let Err(e) = exporter.export(&telemetry_data).await {
                        eprintln!("Failed to export telemetry: {}", e);
                    }
                }
            }
        });
    }

    pub async fn stop(&self) {
        *self.running.write().await = false;
    }

    pub async fn export_now(&self) -> Result<(), Box<dyn std::error::Error>> {
        let telemetry_data = TelemetryData {
            timestamp: Utc::now(),
            metrics: self.metrics_collector.export_metrics(),
            traces: self.tracer.export_traces(),
            costs: {
                let tracker = self.cost_tracker.read().await;
                serde_json::to_value(tracker.generate_report())?
            },
        };

        for exporter in &self.exporters {
            exporter.export(&telemetry_data).await?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryData {
    pub timestamp: DateTime<Utc>,
    pub metrics: serde_json::Value,
    pub traces: serde_json::Value,
    pub costs: serde_json::Value,
}

#[async_trait::async_trait]
pub trait Exporter: Send + Sync {
    async fn export(&self, data: &TelemetryData) -> Result<(), Box<dyn std::error::Error>>;
    fn clone_box(&self) -> Box<dyn Exporter>;
}

/// Console exporter for debugging
pub struct ConsoleExporter;

impl ConsoleExporter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Exporter for ConsoleExporter {
    async fn export(&self, data: &TelemetryData) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Telemetry Export at {} ===", data.timestamp);
        println!("Metrics: {}", serde_json::to_string_pretty(&data.metrics)?);
        println!("Traces: {}", serde_json::to_string_pretty(&data.traces)?);
        println!("Costs: {}", serde_json::to_string_pretty(&data.costs)?);
        println!("===============================\n");
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Exporter> {
        Box::new(Self)
    }
}

/// File exporter
pub struct FileExporter {
    file_path: String,
}

impl FileExporter {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }
}

#[async_trait::async_trait]
impl Exporter for FileExporter {
    async fn export(&self, data: &TelemetryData) -> Result<(), Box<dyn std::error::Error>> {
        use tokio::fs::OpenOptions;
        use tokio::io::AsyncWriteExt;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .await?;

        let json_line = serde_json::to_string(data)? + "\n";
        file.write_all(json_line.as_bytes()).await?;
        file.flush().await?;

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Exporter> {
        Box::new(Self {
            file_path: self.file_path.clone(),
        })
    }
}

/// HTTP exporter
pub struct HttpExporter {
    endpoint: String,
    format: HttpFormat,
    headers: HashMap<String, String>,
    client: reqwest::Client,
}

impl HttpExporter {
    pub fn new(endpoint: String, format: HttpFormat, headers: HashMap<String, String>) -> Self {
        Self {
            endpoint,
            format,
            headers,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl Exporter for HttpExporter {
    async fn export(&self, data: &TelemetryData) -> Result<(), Box<dyn std::error::Error>> {
        let mut request = self.client.post(&self.endpoint);

        // Add headers
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        // Set content type and body based on format
        match self.format {
            HttpFormat::Json => {
                request = request
                    .header("Content-Type", "application/json")
                    .json(data);
            }
            HttpFormat::JsonLines => {
                let json_line = serde_json::to_string(data)?;
                request = request
                    .header("Content-Type", "application/x-ndjson")
                    .body(json_line);
            }
            HttpFormat::Protobuf => {
                // For protobuf, we would need to serialize to protobuf format
                // For now, fall back to JSON
                request = request
                    .header("Content-Type", "application/x-protobuf")
                    .json(data);
            }
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(format!("HTTP export failed with status: {}", response.status()).into());
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Exporter> {
        Box::new(Self {
            endpoint: self.endpoint.clone(),
            format: self.format.clone(),
            headers: self.headers.clone(),
            client: reqwest::Client::new(),
        })
    }
}

/// Jaeger exporter (simplified)
pub struct JaegerExporter {
    endpoint: String,
    client: reqwest::Client,
}

impl JaegerExporter {
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl Exporter for JaegerExporter {
    async fn export(&self, data: &TelemetryData) -> Result<(), Box<dyn std::error::Error>> {
        // Convert our trace format to Jaeger format
        // This is a simplified implementation
        let jaeger_data = serde_json::json!({
            "data": [data.traces],
            "timestamp": data.timestamp
        });

        let response = self
            .client
            .post(format!("{}/api/traces", self.endpoint))
            .header("Content-Type", "application/json")
            .json(&jaeger_data)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Jaeger export failed with status: {}", response.status()).into());
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Exporter> {
        Box::new(Self {
            endpoint: self.endpoint.clone(),
            client: reqwest::Client::new(),
        })
    }
}

/// Prometheus exporter
pub struct PrometheusExporter {
    endpoint: String,
    client: reqwest::Client,
}

impl PrometheusExporter {
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl Exporter for PrometheusExporter {
    async fn export(&self, data: &TelemetryData) -> Result<(), Box<dyn std::error::Error>> {
        // Convert metrics to Prometheus format
        // This is a simplified implementation
        let prometheus_data = self.convert_to_prometheus_format(&data.metrics)?;

        let response = self
            .client
            .post(format!("{}/api/v1/write", self.endpoint))
            .header("Content-Type", "application/x-protobuf")
            .body(prometheus_data)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!(
                "Prometheus export failed with status: {}",
                response.status()
            )
            .into());
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Exporter> {
        Box::new(Self {
            endpoint: self.endpoint.clone(),
            client: reqwest::Client::new(),
        })
    }
}

impl PrometheusExporter {
    fn convert_to_prometheus_format(
        &self,
        _metrics: &serde_json::Value,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // This would convert our metrics to Prometheus text format
        // For now, return a placeholder
        Ok("# Prometheus metrics would go here\n".to_string())
    }
}

/// OpenTelemetry exporter
pub struct OpenTelemetryExporter {
    endpoint: String,
    client: reqwest::Client,
}

impl OpenTelemetryExporter {
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl Exporter for OpenTelemetryExporter {
    async fn export(&self, data: &TelemetryData) -> Result<(), Box<dyn std::error::Error>> {
        // Convert to OpenTelemetry format
        let otel_data = serde_json::json!({
            "resourceSpans": data.traces,
            "resourceMetrics": data.metrics,
            "timestamp": data.timestamp
        });

        let response = self
            .client
            .post(format!("{}/v1/traces", self.endpoint))
            .header("Content-Type", "application/json")
            .json(&otel_data)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!(
                "OpenTelemetry export failed with status: {}",
                response.status()
            )
            .into());
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Exporter> {
        Box::new(Self {
            endpoint: self.endpoint.clone(),
            client: reqwest::Client::new(),
        })
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            export_interval: Duration::from_secs(60),
            exporters: vec![ExporterConfig {
                name: "console".to_string(),
                exporter_type: ExporterType::Console,
                endpoint: None,
                headers: HashMap::new(),
                enabled: false,
            }],
            batch_size: 100,
            max_queue_size: 1000,
        }
    }
}
