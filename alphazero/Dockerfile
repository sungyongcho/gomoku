FROM python:3.13-slim

# Build dependencies for C++ extensions (pybind11/scikit-build-core).
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    ninja-build \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy source before install because scikit-build needs `cpp/` at build time.
COPY . /app
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e ".[serve,torch-cpu]" \
      --extra-index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH=/app
ENV ALPHAZERO_CONFIG=configs/local_play.yaml
ENV ALPHAZERO_CHECKPOINT=models/champion.pt
ENV ALPHAZERO_DEVICE=cpu

EXPOSE 8080
CMD ["uvicorn", "server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
