mod api;
mod config;
mod core;
mod models;
mod workers;

use actix_web::{web, App, HttpServer};
use s3::creds::Credentials;
use s3::{Bucket, Region};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_env("SUBARR_LOG_LEVEL"))
        .init();

    let config = config::Config::from_env();
    let bind_host = config.host.clone();
    let bind_port = config.port;

    let redis_client = redis::Client::open(config.redis_url.as_str())
        .expect("Failed to connect to Redis");

    let region = Region::Custom {
        region: config.s3_region.clone(),
        endpoint: config.s3_endpoint.clone(),
    };
    let credentials = Credentials::new(
        Some(&config.s3_access_key),
        Some(&config.s3_secret_key),
        None,
        None,
        None,
    )
    .expect("Failed to create S3 credentials");

    let s3_bucket = Bucket::new(&config.s3_bucket, region, credentials)
        .expect("Failed to create S3 bucket client")
        .with_path_style();

    let app_state = models::AppState {
        redis: redis_client,
        config,
        s3_bucket,
    };

    tracing::info!("Starting Subarr API on {}:{}", bind_host, bind_port);

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .configure(api::routes::configure)
    })
    .bind((bind_host.as_str(), bind_port))?
    .run()
    .await
}
