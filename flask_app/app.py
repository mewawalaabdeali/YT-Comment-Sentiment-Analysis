import mlflow
from flask import Flask, request, jsonify, send_file
from mlflow.tracking import MlflowClient
import io
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    #Set mlflow tracking URI
    mlflow.set_tracking_uri('http://ec2-3-142-68-25.us-east-2.compute.amazonaws.com:5000/')
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    print(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

#Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error" : "No comments provided"}), 400
    
    try:
        #Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        #Transform comment using vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        X_df = pd.DataFrame(transformed_comments.toarray(),columns=vectorizer.get_feature_names_out()).astype("float64")

        #Make predictions
        predictions = model.predict(X_df).tolist()

        #Convert predictions to strings for consistency
        predictions= [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error":f"Prediction failed: {str(e)}"}), 500
    
    response = [{"comment":comment, "sentiment":sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)