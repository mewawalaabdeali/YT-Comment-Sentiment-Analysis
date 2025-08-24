import os
import mlflow
from mlflow.tracking import MlflowClient
import pytest
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

def promote_model():
    # Set up AWS MLflow tracking URI
    mlflow.set_tracking_uri('http://3.137.209.49:5000/')

    client = mlflow.MlflowClient()

    model_name = "yt_chrome_plugin_model"
    # Get the latest version in staging
    #staging_list = client.get_latest_versions(model_name, stages=["Staging"])

    # --- FIX: index [0] and handle "no staging" case
    staging_list = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_list:
        raise RuntimeError(f"No model found in 'Staging' for '{model_name}'")
    latest_version_staging = staging_list[0].version

    # Archive current Production (if any)
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for mv in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,   # str is fine; MLflow accepts str here
            stage="Archived",
        )

    # Promote Staging -> Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production",
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()