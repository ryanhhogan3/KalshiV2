FROM python:3.11-slim

# System deps for psycopg and SSL
RUN apt-get update \
    && apt-get install -y build-essential libpq-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default environment (override in EC2 task/user-data as needed)
ENV PYTHONUNBUFFERED=1 \
    DB_SSLMODE=require

# Run the DB export script by default
CMD ["python", "src/data/db_export_open_markets.py"]
