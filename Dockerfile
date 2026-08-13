# Base image matches the Python version used throughout local development
# (3.10, confirmed via wheel tags in the original environment notes).
# "slim" keeps the base image small while still being a full CPython
# build (as opposed to "alpine", which uses musl libc and can cause
# subtle wheel-compatibility issues with scientific packages like numpy/
# pandas/scikit-learn).
FROM python:3.10-slim

WORKDIR /app

# Install dependencies BEFORE copying application code. Docker caches
# each instruction as a layer; as long as requirements-prod.txt hasn't
# changed, Docker reuses the cached "pip install" layer on rebuilds
# instead of re-downloading/reinstalling every package, so code-only
# changes (editing app.py) rebuild in seconds, not minutes.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Now copy application code and the trained model. These change far
# more often than dependencies, so they're copied last to maximize
# cache reuse from the layer above.
COPY src/ src/
COPY models/model.joblib models/model.joblib

# Run as a non-root user. If the container were ever compromised via a
# dependency vulnerability, a non-root process limits what an attacker
# can do inside the container — standard container-security practice,
# not specific to this project.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Lets Docker (and orchestration platforms like Render) know whether the
# container is actually serving traffic, not just running — reuses the
# same /health endpoint src/app.py already exposes for exactly this
# purpose.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]