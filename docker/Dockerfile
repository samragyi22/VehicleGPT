FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir llama-cpp-python==0.3.36
WORKDIR /app
COPY llm/grammar_test.py llm/grammar_test.py
CMD ["python", "llm/grammar_test.py"]