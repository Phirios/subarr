use actix_multipart::Multipart;
use actix_web::{web, HttpResponse, get, post};
use chrono::Utc;
use futures_util::StreamExt;
use redis::AsyncCommands;
use uuid::Uuid;

use crate::models::*;
use crate::core::formatter::{format_ass, TranslatedEntry};

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .service(health)
            .service(submit_translation)
            .service(get_job_status)
            .service(download_result)
    );
}

#[get("/health")]
async fn health() -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[post("/translate")]
async fn submit_translation(
    state: web::Data<AppState>,
    mut payload: Multipart,
) -> HttpResponse {
    let job_id = Uuid::new_v4().to_string();

    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut subtitle_bytes: Option<Vec<u8>> = None;
    let mut target_language = "tr".to_string();
    let mut metadata: Option<JobMetadata> = None;
    let mut media_name: Option<String> = None;

    while let Some(Ok(mut field)) = payload.next().await {
        let field_name = field.name().map(|s| s.to_string()).unwrap_or_default();

        match field_name.as_str() {
            "audio" => {
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = field.next().await {
                    bytes.extend_from_slice(&chunk);
                }
                audio_bytes = Some(bytes);
            }
            "subtitle" => {
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = field.next().await {
                    bytes.extend_from_slice(&chunk);
                }
                subtitle_bytes = Some(bytes);
            }
            "target_language" => {
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = field.next().await {
                    bytes.extend_from_slice(&chunk);
                }
                target_language = String::from_utf8_lossy(&bytes).to_string();
            }
            "media_name" => {
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = field.next().await {
                    bytes.extend_from_slice(&chunk);
                }
                media_name = Some(String::from_utf8_lossy(&bytes).to_string());
            }
            "metadata" => {
                let mut bytes = Vec::new();
                while let Some(Ok(chunk)) = field.next().await {
                    bytes.extend_from_slice(&chunk);
                }
                metadata = serde_json::from_slice(&bytes).ok();
            }
            _ => {}
        }
    }

    // Merge media_name into metadata
    if let Some(name) = media_name {
        let meta = metadata.get_or_insert(JobMetadata {
            tmdb_id: None,
            season: None,
            episode: None,
            title: None,
        });
        if meta.title.is_none() {
            meta.title = Some(name);
        }
    }

    let (Some(audio_bytes), Some(subtitle_bytes)) = (audio_bytes, subtitle_bytes) else {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Both 'audio' and 'subtitle' files are required"
        }));
    };

    // Upload to MinIO
    let audio_key = format!("{}/audio", job_id);
    let subtitle_key = format!("{}/subtitle.srt", job_id);

    if let Err(e) = state.s3_bucket.put_object(&audio_key, &audio_bytes).await {
        return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Failed to upload audio to storage: {}", e)
        }));
    }
    if let Err(e) = state.s3_bucket.put_object(&subtitle_key, &subtitle_bytes).await {
        return HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Failed to upload subtitle to storage: {}", e)
        }));
    }

    let now = Utc::now();
    let job = Job {
        id: job_id.clone(),
        status: JobStatus::Queued,
        target_language: target_language.clone(),
        created_at: now,
        updated_at: now,
        error: None,
        result_path: None,
        metadata: metadata.clone(),
    };

    // Store job in Redis
    let mut conn = match state.redis.get_multiplexed_async_connection().await {
        Ok(c) => c,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Redis error: {}", e)
            }));
        }
    };

    let job_json = serde_json::to_string(&job).unwrap();
    let _: () = conn.set(format!("job:{}", job_id), &job_json).await.unwrap();

    // Push task to worker queue (paths are MinIO keys, worker downloads them)
    let task = WorkerTask {
        job_id: job_id.clone(),
        audio_path: audio_key,
        subtitle_path: subtitle_key,
        target_language,
        metadata,
    };
    let task_json = serde_json::to_string(&task).unwrap();
    let _: () = conn.lpush("subarr:tasks", &task_json).await.unwrap();

    HttpResponse::Accepted().json(JobResponse {
        id: job.id,
        status: job.status,
        created_at: job.created_at,
        updated_at: job.updated_at,
        error: None,
    })
}

#[get("/jobs/{job_id}")]
async fn get_job_status(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let job_id = path.into_inner();

    let mut conn = match state.redis.get_multiplexed_async_connection().await {
        Ok(c) => c,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Redis error: {}", e)
            }));
        }
    };

    let job_json: Option<String> = conn.get(format!("job:{}", job_id)).await.unwrap_or(None);

    match job_json {
        Some(json) => {
            let job: Job = serde_json::from_str(&json).unwrap();
            HttpResponse::Ok().json(JobResponse {
                id: job.id,
                status: job.status,
                created_at: job.created_at,
                updated_at: job.updated_at,
                error: job.error,
            })
        }
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Job not found"
        })),
    }
}

#[get("/jobs/{job_id}/download")]
async fn download_result(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let job_id = path.into_inner();

    let mut conn = match state.redis.get_multiplexed_async_connection().await {
        Ok(c) => c,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Redis error: {}", e)
            }));
        }
    };

    let job_json: Option<String> = conn.get(format!("job:{}", job_id)).await.unwrap_or(None);

    match job_json {
        Some(json) => {
            let job: Job = serde_json::from_str(&json).unwrap();
            match job.result_path {
                Some(ref result_key) => {
                    match state.s3_bucket.get_object(result_key).await {
                        Ok(response) => {
                            let json_str = String::from_utf8_lossy(response.bytes());
                            let entries: Vec<TranslatedEntry> = match serde_json::from_str(&json_str) {
                                Ok(e) => e,
                                Err(e) => {
                                    return HttpResponse::InternalServerError().json(serde_json::json!({
                                        "error": format!("Failed to parse result: {}", e)
                                    }));
                                }
                            };
                            let title = job.metadata
                                .as_ref()
                                .and_then(|m| m.title.clone())
                                .unwrap_or_else(|| job_id.clone());
                            let ass_content = format_ass(&entries, &title);
                            HttpResponse::Ok()
                                .content_type("text/x-ssa; charset=utf-8")
                                .append_header(("Content-Disposition", format!("attachment; filename=\"{}.ass\"", job_id)))
                                .body(ass_content)
                        }
                        Err(_) => HttpResponse::NotFound().json(serde_json::json!({
                            "error": "Result file not found in storage"
                        })),
                    }
                }
                None => HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Job not completed yet"
                })),
            }
        }
        None => HttpResponse::NotFound().json(serde_json::json!({
            "error": "Job not found"
        })),
    }
}
