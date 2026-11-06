FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY cctvgpt ./cctvgpt
COPY docs ./docs
RUN pip install --no-cache-dir .
ENV CCTVGPT_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000
CMD ["cctvgpt", "serve", "--host", "0.0.0.0", "--port", "8000"]
