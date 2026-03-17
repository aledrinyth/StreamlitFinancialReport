# Docker Build & Run Guide

## Prerequisites
- Docker installed (v20.10+)
- Docker Compose installed (v1.29+)
- Environment variables configured

### Minimum System Requirements
- 4GB RAM (8GB+ recommended due to large ML models like torch)
- 5GB free disk space
- CPU with at least 2 cores

## Environment Variables

Create a `.env` file in this directory with the following variables:

```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GOOGLE_API_KEY=your_google_api_key
```

You can copy from `.env.example`:
```bash
cp .env.example .env
# Then edit .env with your actual values
```

## ✅ Fixed Issues for Docker Compatibility

This project has been updated with the following fixes:

1. **Added missing Python dependency**: `google-generativeai` (required for Gemini integration)
2. **Fixed hardcoded file paths**: Changed relative path `"dashboard/definitions.json"` to use `os.path.dirname(__file__)` for proper Docker mounting
3. **Environment variable handling**: Updated model.py and text_extracter.py to properly read secrets from environment variables (Docker-compatible)
4. **Flexible cache/results directories**: sentence_model.py now uses environment variables for directory paths
5. **Created Python package marker**: Added `__init__.py` to the `using/` directory for proper module imports
6. **Streamlit headless configuration**: Added proper `.streamlit/config.toml` for containerized operation
7. **Health checks**: Dockerfile includes health check endpoint

## Run with Docker Compose (Recommended)

**Note:** First build may take 5-10 minutes due to downloading large models (torch, transformers, spaCy).

```bash
docker-compose up --build
```

Then access the app at: **http://localhost:8501**

To run in detached mode:
```bash
docker-compose up -d --build
```

View logs:
```bash
docker-compose logs -f dashboard
```

Stop the container:
```bash
docker-compose down
```

## Run with Docker directly

### Build the image
```bash
docker build -t financial-dashboard .
```

### Run the container
```bash
docker run -p 8501:8501 \
  -e SUPABASE_URL="your_supabase_url" \
  -e SUPABASE_KEY="your_supabase_key" \
  -e GOOGLE_API_KEY="your_google_api_key" \
  -v $(pwd)/cache_finbert:/app/cache_finbert \
  -v $(pwd)/results:/app/results \
  financial-dashboard
```

### Or using --env-file
```bash
docker run -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/cache_finbert:/app/cache_finbert \
  -v $(pwd)/results:/app/results \
  financial-dashboard
```

## Troubleshooting

### Port 8501 already in use
Map to a different port:
```bash
docker run -p 8502:8501 financial-dashboard
# Then access at http://localhost:8502
```

### Memory issues / Out of memory (OOM)
The app uses large ML models (torch, transformers). If you get OOM errors:

**Option 1: Increase Docker's memory limit**
- Docker Desktop: Settings → Resources → Memory (increase to 4GB-6GB)
- Docker CLI: `docker run -m 4g financial-dashboard`

**Option 2: Use docker-compose with memory limits** (add to docker-compose.yml)
```yaml
services:
  dashboard:
    mem_limit: 4g
    memswap_limit: 4g
```

### Container exits immediately / Fails to start
Check the logs:
```bash
docker-compose logs dashboard
# or
docker logs <container_id>
```

Common issues:
- Missing `.env` file with API keys
- Insufficient disk space (check with `docker system df`)
- Port 8501 in use by another service
- Internet connectivity issues (needed to download models)

### Models fail to download on first run
If downloading spaCy or NLTK models fails:
```bash
# Rebuild with verbose output
docker-compose up --build
```

The app will retry downloading models on next startup.

### API key errors
Ensure your `.env` file has valid credentials:
- `SUPABASE_URL`: Full URL (e.g., `https://xxxxx.supabase.co`)
- `SUPABASE_KEY`: Valid API key
- `GOOGLE_API_KEY`: Valid Gemini API key

### "404 Not Found" in browser
1. Check container is running: `docker ps`
2. Wait 30-60 seconds for Streamlit to fully start
3. Refresh browser (Ctrl+R or Cmd+R)
4. Check logs: `docker-compose logs`

## Performance Notes

- **First build**: 5-10 minutes (downloads ~2GB of ML models)
- **Startup time**: 30-60 seconds (loading models into memory)
- **Subsequent runs**: 2-3 seconds (models cached)

## Files Created/Modified

- `Dockerfile`: Container configuration
- `.dockerignore`: Excludes unnecessary files from Docker build
- `docker-compose.yml`: Multi-container orchestration
- `.streamlit/config.toml`: Streamlit app configuration for headless mode
- `.env.example`: Template for environment variables
- `using/requirements.txt`: Updated with missing dependencies
- `dashboard.py`: Fixed hardcoded file paths
- `model.py`: Added environment variable support for API keys
- `using/text_extracter.py`: Added environment variable support
- `using/sentence_model.py`: Made cache/results paths configurable
- `using/__init__.py`: Added package marker for proper imports

## Custom volumes

The docker-compose mounts two volumes for persistent data:
- `./cache_finbert:/app/cache_finbert` - Cache for FinBERT model results
- `./results:/app/results` - Output results from analysis

These directories will be created automatically on first run.

## View Logs & Debugging

Stream live logs:
```bash
docker-compose logs -f dashboard --tail 50
```

Debug Streamlit startup:
```bash
docker run -e STREAMLIT_LOGGER_LEVEL=debug -p 8501:8501 financial-dashboard
```

## Clean up

Remove stopped containers:
```bash
docker-compose down
```

Remove all images:
```bash
docker-compose down --rmi all
```

Remove unused Docker resources:
```bash
docker system prune -a
```

