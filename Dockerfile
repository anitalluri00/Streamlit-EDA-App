# Use official python slim image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements and app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Expose the streamlit port
EXPOSE 8501

# Streamlit config: run headless
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
