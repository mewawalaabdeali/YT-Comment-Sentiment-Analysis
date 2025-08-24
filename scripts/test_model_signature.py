import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import pickle


# Set your remote tracking URI
mlflow.set_tracking_uri('http://3.137.209.49:5000/')

@pytest.mark.parametrize("model_name, alias, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),
    ])


def test_model_with_vectorizer(model_name, alias, vectorizer_path):
    client = MlflowClient()
    
    # Get the latest version in the specified stage
    latest_version_info = client.get_model_version_by_alias(model_name, alias)
    latest_version = int(latest_version_info.version) if latest_version_info else None
    
    assert latest_version is not None, f"No model found in the '{alias}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        #load vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        #Create a dummy input for the model
        input_text = 'I liked the content of your video. Well researched.'
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_features_names_out())

        prediction = model.predict(input_df)

        assert input_df.shape[1] == len(vectorizer.get_features_names_out()), "Input feature count mismatch"

        #Verify the output shape(assuming binary classification with a single output)
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version} successfully processed the dummy input text.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")