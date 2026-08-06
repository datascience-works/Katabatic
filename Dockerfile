FROM python:3.11-slim

ARG POETRY_INSTALL_ARGS="--only main"
ARG MODEL_EXTRA=""

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
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy dependency files
COPY pyproject.toml poetry.lock README.md LICENSE ./

# Copy Katabatic source code
COPY katabatic ./katabatic

# Pre-install CPU-only torch.
RUN if echo "${MODEL_EXTRA}" | grep -q "great"; then \
        pip install --no-cache-dir "torch>=2.7.1,<3.0.0" \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi
#Specifies which extras to install or just main.
RUN poetry install ${POETRY_INSTALL_ARGS} ${MODEL_EXTRA}

#Non-root user for runtime
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Verify installation
RUN python -c "import katabatic; print('Katabatic version:', katabatic.__version__)"

# Default command
CMD ["python", "-c", "import katabatic; print('Katabatic is installed and ready')"]