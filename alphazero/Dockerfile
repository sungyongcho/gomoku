# Use the official Python image
FROM python:3.13-slim

# Prevent stdout/stderr buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt /app/requirements.txt
COPY requirements.cuda.txt /app/requirements.cuda.txt
COPY requirements.cpu.txt /app/requirements.cpu.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/requirements.cpu.txt

# Copy the application code
COPY . .

# Add PYTHONPATH to ensure imports work
ENV PYTHONPATH=/app

# Expose any port; actual port is controlled by env var
# default FastAPI port (can be overridden)
ENV LOCAL_ALPHAZERO=8000
EXPOSE ${LOCAL_ALPHAZERO}

# Use a shell command to use the env var
CMD ["sh", "-c", "uvicorn adapters.api_fastapi:create_app --factory --host 0.0.0.0 --port ${LOCAL_ALPHAZERO} --reload"]
