FROM python:3.10-slim

# Install build-essential & libpq-dev for some Python deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
 && apt-get clean

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
