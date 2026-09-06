FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sonic_server.py .

EXPOSE 8001

CMD ["uvicorn", "sonic_server:app", "--host", "0.0.0.0", "--port", "8001"]
