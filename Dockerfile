FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
# Add extra reqs that might not be in base txt
RUN pip install --no-cache-dir -r requirements.txt peft pandas pyyaml streamlit

# Copy source
COPY . .

# Install package
RUN pip install -e .

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit
EXPOSE 8501

# Default command (UI)
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
