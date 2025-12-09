FROM python:3.9-slim

# Dependencias sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer puertos para backend (5000) y frontend (8501)
EXPOSE 5000
EXPOSE 8501

# CMD por defecto (docker-compose sobrescribirá)
CMD ["bash"]