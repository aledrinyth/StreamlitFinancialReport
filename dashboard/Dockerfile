FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel, and packaging FIRST
RUN pip install --upgrade pip setuptools wheel packaging importlib-metadata

# Copy requirements
COPY using/requirements.txt requirements.txt

# Install Python dependencies with proper order
RUN pip install --no-cache-dir setuptools>=68.0.0 && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary NLP models
RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader punkt averaged_perceptron_tagger 2>/dev/null || true

# Copy application files
COPY . .

# Create Streamlit config
RUN mkdir -p /app/.streamlit && cat > /app/.streamlit/config.toml << 'EOF'
[client]
showErrorDetails = true

[logger]
level = "warning"

[server]
port = 8501
headless = true
runOnSave = false
enableCORS = false
enableXsrfProtection = true
address = "0.0.0.0"

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[runner]
magicEnabled = true
EOF

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "dashboard.py", "--logger.level=debug", "--client.showErrorDetails=true"]
