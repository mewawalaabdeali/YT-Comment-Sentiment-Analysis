import mlflow
from flask import Flask, request, jsonify, send_file
from mlflow.tracking import MlflowClient

#Set mlflow tracking URI
mlflow.set_tracking_uri('http://ec2-3-17-37-181.us-east-2.compute.amazonaws.com:5000/')

def load_model_from_registry(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    print(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_model_from_registry("yt_chrome_plugin_model", "2")
print("Model loaded succesfully")