# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps (if required for some packages)
RUN apt-get update && apt-get install -y build-essential wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port=8501", "--server.address=0.0.0.0"]
