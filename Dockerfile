# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_dashboard.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_dashboard.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application files
COPY email_classifier_dashboard.py .
COPY train_models.py .
COPY simple_test.py .
COPY setup.py .

# Copy data files (if they exist)
COPY emails_cleaned.csv . 2>/dev/null || true
COPY *.pkl . 2>/dev/null || true

# Expose port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "email_classifier_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
