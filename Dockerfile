FROM python:3.10-slim

# Install Java Runtime for tabula-py (PDF support) and build-essentials for dependencies
RUN apt-get update && apt-get install -y \
    default-jre \
    build-essential \
    libpq-dev \
 && apt-get clean

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
