from dagster import execute_job
from src.main import ml_pipeline

if __name__ == "__main__":
    result = execute_job(ml_pipeline)
    if result.success:
        print("Pipeline succeeded!")
    else:
        print("Pipeline failed!")