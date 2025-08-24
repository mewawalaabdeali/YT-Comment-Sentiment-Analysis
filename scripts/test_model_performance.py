import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set your remote tracking URI
mlflow.set_tracking_uri('http://3.137.209.49:5000/')

@pytest.mark.parametrize("model_name, alias, holdout_data_path, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),
    ])


def test_model_with_vectorizer(model_name, alias, holdout_data_path, vectorizer_path):
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

        #Load the holdout test data , prepare input and get prediction
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, :-1].squeeze()
        y_holdout = holdout_data.iloc[:,-1]

        X_holdout_raw = X_holdout_raw.fillna("")

        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(X_holdout_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

        y_pred_new = model.predict(X_holdout_tfidf_df)

        #Calculate performance metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        assert accuracy_new >= expected_accuracy, f'Accuracy should be at least {expected_accuracy}, got {accuracy_new}'
        assert precision_new >= expected_precision, f'Precision should be at least {expected_precision}, got {precision_new}'
        assert recall_new >= expected_recall, f'Recall should be at least {expected_recall}, got {recall_new}'
        assert f1_new >= expected_f1, f'F1 score should be at least {expected_f1}, got {f1_new}'

        print(f"Performance test passed for model '{model_name}' version {latest_version}")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")