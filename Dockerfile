FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data/raw ./data/raw
ENV DCF_PROJECT_ROOT=/app
# Build bronze -> silver -> gold and train at image build time (about 5 seconds).
RUN python -m dcforecast.pipeline

EXPOSE 8000
CMD ["sh", "-c", "uvicorn dcforecast.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
