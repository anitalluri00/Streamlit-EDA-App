FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

# Install dependencies including xlrd and openpyxl
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
