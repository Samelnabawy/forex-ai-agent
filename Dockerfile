FROM python:3.12-slim

WORKDIR /app

# System deps for pandas-ta / numpy / scipy
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

EXPOSE 8000 9100

CMD ["uvicorn", "src.monitoring.dashboard:app", "--host", "0.0.0.0", "--port", "8000"]
