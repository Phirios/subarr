use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    // Critical - from env only
    pub redis_url: String,
    pub gemini_api_key: String,
    pub tmdb_api_key: Option<String>,
    pub pyannote_auth_token: String,

    // Server
    pub host: String,
    pub port: u16,

    // MinIO/S3 storage
    pub s3_endpoint: String,
    pub s3_access_key: String,
    pub s3_secret_key: String,
    pub s3_bucket: String,
    pub s3_region: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            redis_url: env::var("SUBARR_REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379/0".to_string()),
            gemini_api_key: env::var("SUBARR_GEMINI_API_KEY")
                .expect("SUBARR_GEMINI_API_KEY is required"),
            tmdb_api_key: env::var("SUBARR_TMDB_API_KEY").ok(),
            pyannote_auth_token: env::var("SUBARR_PYANNOTE_AUTH_TOKEN")
                .expect("SUBARR_PYANNOTE_AUTH_TOKEN is required"),
            host: env::var("SUBARR_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("SUBARR_PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .expect("SUBARR_PORT must be a valid port number"),
            s3_endpoint: env::var("SUBARR_S3_ENDPOINT")
                .unwrap_or_else(|_| "http://minio-service.phirios.svc.cluster.local:9000".to_string()),
            s3_access_key: env::var("SUBARR_S3_ACCESS_KEY")
                .unwrap_or_else(|_| "phirios".to_string()),
            s3_secret_key: env::var("SUBARR_S3_SECRET_KEY")
                .unwrap_or_else(|_| "changeme".to_string()),
            s3_bucket: env::var("SUBARR_S3_BUCKET")
                .unwrap_or_else(|_| "subarr".to_string()),
            s3_region: env::var("SUBARR_S3_REGION")
                .unwrap_or_else(|_| "us-east-1".to_string()),
        }
    }
}
