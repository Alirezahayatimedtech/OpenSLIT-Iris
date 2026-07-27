FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/OpenSLIT-Iris
COPY . .
RUN python -m pip install --upgrade pip && \
    python -m pip install -e '.[ai,cvat,google]'

CMD ["bash"]
