FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# El modelo NO se copia a la imagen: se descarga del Model Registry por alias.
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MODEL_NAME=adult-income-classifier \
    MODEL_ALIAS=champion

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
