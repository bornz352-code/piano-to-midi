FROM python:3.10-slim

# System deps: ffmpeg for audio decoding, libsndfile for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir watchdog

# Copy application code
COPY . .

# Create default volume mount points
RUN mkdir -p /input /output

EXPOSE 5050

# Default: run the web server. Override with watchdog_service.py in compose.
CMD ["python", "main.py"]
