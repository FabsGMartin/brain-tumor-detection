#!/bin/bash
# No usar set -e para capturar todos los errores

# Configurar logging
export PYTHONUNBUFFERED=1
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}

echo "=========================================="
echo "Iniciando aplicación Flask"
echo "LOG_LEVEL: $LOG_LEVEL"
echo "PYTHONUNBUFFERED: $PYTHONUNBUFFERED"
echo "PORT: ${PORT:-5000}"
echo "=========================================="

# Ejecutar gunicorn con captura de errores
# Usar exec para que gunicorn sea el proceso principal
exec gunicorn \
    --bind 0.0.0.0:${PORT:-5000} \
    --workers 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level debug \
    --capture-output \
    --enable-stdio-inheritance \
    src.backend.app:app
