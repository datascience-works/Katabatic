FROM python:3.11-slim

ARG POETRY_INSTALL_ARGS="--only main"

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Set working directory
WORKDIR /app

# No build toolchain required. All dependencies resolve to pre-built wheels.

# Install Poetry
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Pre-install CPU-only torch.
RUN if [ -n "${MODEL_EXTRA}" ]; then \
        pip install --no-cache-dir "torch>=2.7.1,<3.0.0" \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Copy dependency files
COPY pyproject.toml poetry.lock README.md LICENSE ./

# Only install dependencies
RUN poetry install ${POETRY_INSTALL_ARGS} ${MODEL_EXTRA} --no-root

# Copy Katabatic source code
COPY katabatic ./katabatic

# Install Katabatic package
RUN poetry install --only-root

# Non-root user for runtime
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Verify installation as runtime user
RUN python -c "import katabatic; print('Katabatic version:', katabatic.__version__)"

# Default command
CMD ["python", "-c", "import katabatic; print('Katabatic is installed and ready')"]
