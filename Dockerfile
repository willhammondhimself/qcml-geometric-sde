FROM python:3.11-slim

WORKDIR /app

# System dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install package in development mode
RUN pip install --no-cache-dir -e .

# Default: run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--ignore=tests/test_crisis_validation.py", "-x"]
