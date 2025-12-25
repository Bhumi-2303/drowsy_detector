# ===============================
# Base Image (Stable for dlib)
# ===============================
FROM python:3.10-slim

# ===============================
# Environment
# ===============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ===============================
# System Dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Working Directory
# ===============================
WORKDIR /app

# ===============================
# Install Python Dependencies
# ===============================
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy Project Files
# ===============================
COPY . .

# ===============================
# Expose Streamlit Port
# ===============================
EXPOSE 8501

# ===============================
# Default Command (Headless-safe)
# ===============================
CMD ["python", "main.py"]
