"""
Subarr ML Worker
Listens on Redis queue "subarr:tasks", processes ML pipeline, pushes results.
"""

import json
import logging
import redis
from datetime import datetime, timezone

from config import settings
from pipeline import Pipeline

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("subarr-worker")


def main():
    r = redis.from_url(settings.redis_url)
    pipe = Pipeline(settings)

    logger.info("Subarr ML Worker started, waiting for tasks...")

    while True:
        # Blocking pop from task queue
        _, task_json = r.brpop("subarr:tasks")
        task = json.loads(task_json)
        job_id = task["job_id"]

        logger.info(f"Processing job {job_id}")

        try:
            result = pipe.process(task, r)

            # Update job as completed
            job = json.loads(r.get(f"job:{job_id}"))
            job["status"] = "completed"
            job["result_path"] = result["output_path"]
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            r.set(f"job:{job_id}", json.dumps(job))

            logger.info(f"Job {job_id} completed")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job = json.loads(r.get(f"job:{job_id}"))
            job["status"] = "failed"
            job["error"] = str(e)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            r.set(f"job:{job_id}", json.dumps(job))


if __name__ == "__main__":
    main()
