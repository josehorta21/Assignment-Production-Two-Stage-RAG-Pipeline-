FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.model_cache
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
RUN uv sync --no-dev --no-install-project
COPY src ./src
COPY docs ./docs
COPY scripts ./scripts
COPY .env.example ./
EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
