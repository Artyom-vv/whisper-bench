FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Сначала зависимости, чтобы кешировалось
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Потом только исходники (уже отфильтрованные .dockerignore)
COPY . .

CMD ["python3", "run.py"]
