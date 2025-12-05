FROM python:3.11-slim

# Evitar problemas de logs y buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalamos dependencias del sistema que necesita opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# puerto interno de la app
ENV PORT=8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
