# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a non-root user for security and pip compliance
RUN useradd -m astroai_user
USER astroai_user

# Ensure user's local bin is in PATH
ENV PATH="/home/astroai_user/.local/bin:${PATH}"

# Copy dependencies first for caching
COPY --chown=astroai_user:astroai_user requirements.txt .

# Install dependencies into user space
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=astroai_user:astroai_user . .

# Expose port
EXPOSE 8000

# Start command using gunicorn for Flask
CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:${PORT:-8000}"]
