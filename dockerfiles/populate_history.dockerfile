FROM python:3.12-slim AS builder

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy requirements and setup files first for better caching
COPY requirements.txt .
COPY setup.py .

# Install dependencies using uv with caching
# First install dependencies without the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy the rest of the code
COPY . .

# Install the project in development mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .

# Compile Python bytecode
RUN python -m compileall -q /app

# Create final image
FROM python:3.12-slim

# Copy uv from builder stage
COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy installed packages and code from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Set environment variables
# GCP_CREDENTIALS should be injected at runtime or through docker secrets
ENV API_PORT=8000
ENV PYTHONPATH=/app
ENV UV_SYSTEM_PYTHON=1

# Expose the API port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "src.history.set_history_items"]