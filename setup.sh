#!/bin/bash
# Alpha Flow — one-time environment setup
# M0-M4 are entirely free (paper trading, free APIs, local GPU)

set -e

# 1. Python environment
conda create -n alphaflow python=3.11 -y
conda activate alphaflow

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# JAX (for DGM HJB solver)
pip install "jax[cuda12]" flax optax
# ML + NLP
pip install transformers sentence-transformers
# Data + infra
pip install kafka-python psycopg2-binary sqlalchemy
pip install alpaca-trade-api polygon-api-client fredapi newsapi-python
# Orchestration + visualization
pip install apache-airflow streamlit plotly pandas numpy scipy
# Dev tools
pip install pytest black mypy

# 2. TimescaleDB (Docker)
docker run -d --name timescaledb -p 5432:5432 \
    -e POSTGRES_PASSWORD=alphaflow \
    timescale/timescaledb:latest-pg15

# 3. Kafka (Docker)
docker run -d --name kafka -p 9092:9092 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    confluentinc/cp-kafka:latest

# 4. Initialize DB schema
psql postgresql://postgres:alphaflow@localhost:5432/alphaflow \
    -f data/schema/timescale.sql

# 5. Create .env (fill in keys when ready)
cat > .env << 'EOF'
POLYGON_KEY=[YOUR_KEY_HERE]
ALPACA_KEY=[YOUR_KEY_HERE]
ALPACA_SECRET=[YOUR_SECRET_HERE]
FRED_KEY=[YOUR_KEY_HERE]
NEWSAPI_KEY=[YOUR_KEY_HERE]
BARCHART_KEY=[YOUR_KEY_HERE]
DB_URL=postgresql://postgres:alphaflow@localhost:5432/alphaflow
KAFKA_BOOTSTRAP=localhost:9092
EOF

echo ""
echo "Setup complete."
echo "  - Fill .env with real keys when ready (FRED is free; Alpaca paper is free)"
echo "  - Paper trading works immediately with free Alpaca account"
echo "  - M0-M4 milestone cost: \$0"
