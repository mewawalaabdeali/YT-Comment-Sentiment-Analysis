import os
import mlflow
from mlflow.tracking import MlflowClient
import pytest
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

model_name = "yt_chrome_plugin_model"
from_alias = "staging"
to_alias = "production"

def promote_model(model_name, from_alias, to_alias):
    # Set your remote tracking URI
    mlflow.set_tracking_uri('http://3.134.94.213:5000/')

    client = MlflowClient()

    
    
    try:
        model_staging = client.get_model_version_by_alias(model_name, from_alias)
    except MlflowException as e:
        raise RuntimeError(f"Alias '{from_alias}' not found for '{model_name}': {e}")
    latest_version = int(model_staging.version)

    older_version = None
    try:
        model_production = client.get_model_version_by_alias(model_name, to_alias)
        older_version = int(model_production.version)
    except MlflowException:
        pass

    client.set_registered_model_alias(model_name, to_alias, int(latest_version))

    try:
        if older_version is not None:
            client.transition_model_version_stage(
                name=model_name, version=str(older_version), stage="Archived"
            )
            client.transition_model_version_stage(
                name=model_name, version=str(latest_version), stage="Production"
            )
    except Exception as e:
        print(f"Couldn't promote model: {e}")

    print(f"Promoted {model_name} v{latest_version} -> @{to_alias}"
          + (f"(previous prod was v{older_version}, archived)" if older_version is not None else ""))
    

if __name__=="__main__":
    promote_model("yt_chrome_plugin_model", "staging", "production")

