# HuggingFace Spaces Docker – Customer Support Ticket Resolver
# Docs: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install uv for lock-file based installs
RUN pip install --no-cache-dir uv

# Install dependencies from pyproject.toml
COPY --chown=user ./pyproject.toml pyproject.toml
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY --chown=user . /app

# Expose HF Spaces port
EXPOSE 7860

# Start FastAPI server via server.app package
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
