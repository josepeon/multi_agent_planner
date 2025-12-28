# Multi-Agent Planner Docker Image
# =================================
# Build: docker build -t multi-agent-planner .
# Run:   docker run -p 8080:8080 --env-file .env multi-agent-planner

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/ ./agents/
COPY core/ ./core/
COPY web/ ./web/
COPY main.py .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p output memory logs

# Expose port for web UI
EXPOSE 8080

# Default command: run web server
CMD ["python", "web/app.py"]

# Alternative commands:
# CLI mode: docker run -it multi-agent-planner python main.py
# Custom:   docker run multi-agent-planner python -c "..."
