# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a non-root user
RUN useradd -m astroai_user

# Pre-create necessary directories and set permissions
RUN mkdir -p uploads static/processed static/training_assets && \
    chown -R astroai_user:astroai_user /app

# Switch to non-root user
USER astroai_user

# Ensure user's local bin is in PATH
ENV PATH="/home/astroai_user/.local/bin:${PATH}"

# Copy and install dependencies
COPY --chown=astroai_user:astroai_user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=astroai_user:astroai_user . .

# Expose port
EXPOSE 8000

# Start command using gunicorn
CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120"]
