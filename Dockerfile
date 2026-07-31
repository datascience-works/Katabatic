FROM python:3.11-slim

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install core dependencies only
RUN poetry install --only main --no-root

# Copy Katabatic source code
COPY katabatic ./katabatic

# Verify installation
RUN python -c "import katabatic; print('Katabatic version:', katabatic.__version__)"

# Default command
CMD ["python", "-c", "import katabatic; print('Katabatic is installed and ready')"]
