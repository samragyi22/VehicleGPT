FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" FORCE_CMAKE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY config.py schema.py ./
COPY engine ./engine
COPY llm ./llm
COPY eval ./eval
COPY data/vehicle.db ./data/vehicle.db
RUN pip install --no-cache-dir 'llama-cpp-python==0.3.36' .
CMD ["python", "-m", "eval.benchmark"]
