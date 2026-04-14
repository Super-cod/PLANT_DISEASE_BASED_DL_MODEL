FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 --retries 50 --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py labels.txt model_updated.pth /app/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
