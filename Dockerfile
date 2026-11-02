
# ==========================================
# Agentic Chatbot - Docker Configuration
# ==========================================

FROM python:3.11-slim

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Application directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy application source code
COPY . .

# Create directories for uploaded files and vector index
RUN mkdir -p /app/uploads /app/faiss_index

# Streamlit default port
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "Finalapp.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
