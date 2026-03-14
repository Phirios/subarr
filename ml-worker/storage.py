"""
MinIO/S3 storage client for downloading job files.
"""

import logging
import os
import tempfile
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger("subarr-worker")


class StorageClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str = "subarr"):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def download(self, remote_path: str, local_path: str):
        """Download a file from MinIO to local path."""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info(f"Downloading {self.bucket}/{remote_path} -> {local_path}")
        self.client.download_file(self.bucket, remote_path, local_path)

    def download_job_files(self, job_id: str, work_dir: str) -> tuple[str, str]:
        """Download audio and subtitle for a job. Returns (audio_path, subtitle_path)."""
        audio_path = os.path.join(work_dir, "audio")
        subtitle_path = os.path.join(work_dir, "subtitle.srt")

        self.download(f"{job_id}/audio", audio_path)
        self.download(f"{job_id}/subtitle.srt", subtitle_path)

        return audio_path, subtitle_path

    def upload(self, local_path: str, remote_path: str):
        """Upload a file to MinIO."""
        logger.info(f"Uploading {local_path} -> {self.bucket}/{remote_path}")
        self.client.upload_file(local_path, self.bucket, remote_path)
