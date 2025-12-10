# Dockerfile para desarrollo local (usado por docker-compose)
FROM python:3.9-slim

# Dependencias sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias
# Para desarrollo local, instalamos ambos sets de dependencias
COPY requirements-backend.txt requirements-frontend.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-backend.txt -r requirements-frontend.txt

# Copiar todo el proyecto
COPY . .

# Exponer puertos para backend (5000) y frontend (8501)
EXPOSE 5000
EXPOSE 8501

# CMD por defecto (docker-compose sobrescribirá)
CMD ["bash"]