FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY flask_app/ /app/

COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# ---- Gunicorn (bind to all interfaces; longer timeout for MLflow/S3 load) ----
# If your Flask app is in app.py with `app = Flask(__name__)`, keep "app:app".
# If it's elsewhere, change the module path (e.g., "src.app:app").
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "--timeout", "180",\
     "--access-logfile", "-",\
     "--error-logfile", "-",\
     "--preload",\
     "app:app"]