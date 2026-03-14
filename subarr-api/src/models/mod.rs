use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Clone)]
pub struct AppState {
    pub redis: redis::Client,
    pub config: crate::config::Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Diarizing,
    DetectingEmotion,
    Translating,
    Formatting,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub status: JobStatus,
    pub target_language: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error: Option<String>,
    pub result_path: Option<String>,
    pub metadata: Option<JobMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetadata {
    pub tmdb_id: Option<u64>,
    pub season: Option<u32>,
    pub episode: Option<u32>,
    pub title: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub id: String,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Message sent to Python ML worker via Redis queue
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerTask {
    pub job_id: String,
    pub audio_path: String,
    pub subtitle_path: String,
    pub target_language: String,
    pub metadata: Option<JobMetadata>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_status_serialization() {
        let status = JobStatus::Queued;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"queued\"");

        let status = JobStatus::DetectingEmotion;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"detecting_emotion\"");
    }

    #[test]
    fn test_job_status_deserialization() {
        let status: JobStatus = serde_json::from_str("\"completed\"").unwrap();
        assert!(matches!(status, JobStatus::Completed));

        let status: JobStatus = serde_json::from_str("\"failed\"").unwrap();
        assert!(matches!(status, JobStatus::Failed));
    }

    #[test]
    fn test_job_serialization_roundtrip() {
        let job = Job {
            id: "test-123".to_string(),
            status: JobStatus::Queued,
            target_language: "tr".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            error: None,
            result_path: None,
            metadata: Some(JobMetadata {
                tmdb_id: Some(12345),
                season: Some(1),
                episode: Some(3),
                title: Some("Test Show".to_string()),
            }),
        };

        let json = serde_json::to_string(&job).unwrap();
        let deserialized: Job = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-123");
        assert_eq!(deserialized.target_language, "tr");
        assert!(matches!(deserialized.status, JobStatus::Queued));
        assert_eq!(deserialized.metadata.unwrap().tmdb_id, Some(12345));
    }

    #[test]
    fn test_worker_task_serialization() {
        let task = WorkerTask {
            job_id: "job-1".to_string(),
            audio_path: "/tmp/audio.wav".to_string(),
            subtitle_path: "/tmp/sub.srt".to_string(),
            target_language: "tr".to_string(),
            metadata: None,
        };

        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("job-1"));
        assert!(json.contains("/tmp/audio.wav"));

        let deserialized: WorkerTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.job_id, "job-1");
        assert!(deserialized.metadata.is_none());
    }

    #[test]
    fn test_job_metadata_optional_fields() {
        let meta: JobMetadata = serde_json::from_str("{}").unwrap();
        assert!(meta.tmdb_id.is_none());
        assert!(meta.season.is_none());
        assert!(meta.episode.is_none());
        assert!(meta.title.is_none());
    }
}
