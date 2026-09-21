FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and ensure logs are immediately visible.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by Python packages
# and document processing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for Docker layer caching.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend application.
COPY backend ./backend

# Copy offline evaluation resources.
COPY evaluation ./evaluation

# Runtime directories.
RUN mkdir -p \
    /app/.chroma \
    /app/uploads \
    /app/data

EXPOSE 8000

# Container health check.
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.api:api", "--host", "0.0.0.0", "--port", "8000"]