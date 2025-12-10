#!/bin/bash
# Script simplificado para debugging

# Forzar salida inmediata
export PYTHONUNBUFFERED=1
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}

# Escribir directamente a stderr
echo "=== INICIANDO APLICACIÓN ===" >&2
echo "PORT: ${PORT:-5000}" >&2
echo "LOG_LEVEL: $LOG_LEVEL" >&2

# Ejecutar Python directamente con gunicorn
exec python -m gunicorn \
    --bind 0.0.0.0:${PORT:-5000} \
    --workers 1 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level debug \
    src.backend.app:app
