FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/panns_data && curl -sL -o /root/panns_data/class_labels_indices.csv http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

COPY classifier.py ./tools/classifier.py
COPY isolator.py ./tools/isolator.py
COPY fourier_fundamentals.py ./tools/fourier_fundamentals.py
COPY monitor.py ./tools/monitor.py

COPY fourier_explorer.html ./explorer/fourier_explorer.html
COPY static/ ./explorer/static/

COPY models/ ./models/

COPY backend/app/ ./app/

ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
