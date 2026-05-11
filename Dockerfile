FROM python:3.10-slim

# System deps required by opencv-python-headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching — only rebuilds when requirements change)
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy only what the server needs (skip mediapipe, .venv, data/)
COPY src/__init__.py src/predictor.py src/renderer.py src/
COPY model/model.h5 model/model.h5
COPY server/ server/

# TensorFlow tuning for low-memory CPU environment (Render free tier = 512 MB)
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_NUM_INTRAOP_THREADS=2
ENV TF_NUM_INTEROP_THREADS=1
ENV MALLOC_TRIM_THRESHOLD_=65536
ENV PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["python", "server/api.py"]
