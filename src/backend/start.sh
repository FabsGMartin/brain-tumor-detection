#!/bin/bash
# Forzar salida inmediata de logs
set -x
exec 1>&2  # Redirigir stdout a stderr para asegurar que se vea

# Configurar logging
export PYTHONUNBUFFERED=1
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}

echo "==========================================" >&2
echo "SCRIPT DE INICIO EJECUTÁNDOSE" >&2
echo "LOG_LEVEL: $LOG_LEVEL" >&2
echo "PYTHONUNBUFFERED: $PYTHONUNBUFFERED" >&2
echo "PORT: ${PORT:-5000}" >&2
echo "PWD: $(pwd)" >&2
echo "LS /app:" >&2
ls -la /app >&2 || true
echo "LS /app/src/backend:" >&2
ls -la /app/src/backend >&2 || true
echo "==========================================" >&2

# Verificar que Python está disponible
echo "Verificando Python..." >&2
python --version >&2 || { echo "ERROR: Python no encontrado" >&2; exit 1; }

# Verificar que gunicorn está instalado
echo "Verificando gunicorn..." >&2
gunicorn --version >&2 || { echo "ERROR: gunicorn no encontrado" >&2; exit 1; }

# Verificar que el módulo existe
echo "Verificando módulo src.backend.app..." >&2
python -c "import src.backend.app" >&2 || { echo "ERROR: No se puede importar src.backend.app" >&2; exit 1; }

echo "Iniciando gunicorn..." >&2

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
    src.backend.app:app 2>&1
