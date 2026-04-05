# run_training.py
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import custom_job as gca_custom_job_compat

# Initialize
aiplatform.init(
    project="apollousa",
    location="asia-southeast1",
    staging_bucket="gs://apollousa-sg-models/staging"  # GCS bucket for staging
)

# Create the job from your local script
job = aiplatform.CustomJob.from_local_script(
    display_name="apollousa-training-a100",
    script_path="trainer/task.py",  # Path to your training script
    container_uri="asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
    requirements=[
        'lightning>=2.0.0',           # ← This is critical
        'litdata==0.2.61',
        'huggingface-hub>=0.16.0',
        'wandb>=0.15.0',
        'google-cloud-storage>=2.8.0',
        'gcsfs>=2023.12.0',
        'fsspec>=2023.12.0'
     ],  # Already in setup.py, can leave empty,
     machine_type="a2-highgpu-1g",
     accelerator_type="NVIDIA_TESLA_A100",
     accelerator_count=1,
     boot_disk_size_gb=200,      # ← add this
     boot_disk_type="pd-ssd",
     environment_variables={  # ← This is the correct parameter
        "LIGHTNING_CLOUD_RESOLVE_DIR": "/tmp/litdata_cache"
    }
)

job.job_spec.scheduling.strategy = gca_custom_job_compat.Scheduling.Strategy.SPOT
job.job_spec.scheduling.restart_job_on_worker_restart = True

job.run()