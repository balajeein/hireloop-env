FROM python:3.10.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies with robust retry/timeout settings
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    --index-url https://pypi.org/simple \
    -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=2)" || exit 1

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]