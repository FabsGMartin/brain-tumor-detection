import numpy as np
import os
import base64
import random
import glob
import logging
import sys
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import cv2
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Configurar logging TEMPRANO para capturar errores de inicialización
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("Iniciando aplicación Flask - Brain Tumor Detection")
logger.info(f"Nivel de logging: {log_level}")
logger.info("=" * 60)

# Cargar variables de entorno
try:
    load_dotenv()
    logger.debug("Variables de entorno cargadas desde .env")
except Exception as e:
    logger.warning(f"No se pudo cargar .env: {e}")

# Importar módulos locales con manejo de errores
try:
    logger.debug("Importando módulos locales...")
    from src.backend.model import model_clasificacion, model_segmentacion

    logger.debug("Modelos importados correctamente")
    from src.backend.storage import S3PredictionStorage

    logger.debug("Storage importado correctamente")
except Exception as e:
    logger.error(f"Error importando módulos locales: {e}", exc_info=True)
    raise

# Inicializamos la aplicación Flask
try:
    logger.debug("Creando instancia de Flask...")
    app = Flask(__name__)
    application = app
    logger.debug("Instancia de Flask creada correctamente")
except Exception as e:
    logger.error(f"Error creando instancia Flask: {e}", exc_info=True)
    raise

# Configurar CORS
try:
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    CORS(app, origins=cors_origins, supports_credentials=True)
    logger.info(f"CORS configurado para origins: {cors_origins}")
except Exception as e:
    logger.error(f"Error configurando CORS: {e}", exc_info=True)
    raise

# Configuración S3
# Ahora (busca uno o el otro):
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("BUCKET_NAME")
S3_DATA_PREFIX = os.getenv("S3_DATA_PREFIX", "data/")
S3_PREDICTIONS_PREFIX = os.getenv("S3_PREDICTIONS_PREFIX", "predictions/")

# ---------- PATH CONSTANTS ----------
BASE_DIR = Path(__file__).resolve().parent

# Constantes desde variables de entorno
target_size_str = os.getenv("TARGET_SIZE", "256,256")
TARGET_SIZE = tuple(map(int, target_size_str.split(",")))
LABELS = ["No detectado (0)", "Detectado(1)"]

# Configurar cliente S3 si está disponible
s3_client = None
storage = None
if S3_BUCKET:
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if aws_access_key and aws_secret_key:
            # Desarrollo local: usar credenciales explícitas
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-3"),
            )
            logger.info("Cliente S3 configurado con credenciales explícitas")
        else:
            # Producción (App Runner): usar IAM role
            s3_client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-3"),
            )
            logger.info(
                "Cliente S3 configurado con IAM role (credenciales automáticas)"
            )

        logger.info(f"Cliente S3 configurado para bucket: {S3_BUCKET}")

        # Inicializar almacenamiento de predicciones
        storage = S3PredictionStorage(
            s3_client=s3_client,
            bucket_name=S3_BUCKET,
            prefix=S3_PREDICTIONS_PREFIX,
        )
        logger.info(
            f"Almacenamiento de predicciones configurado con prefijo: {S3_PREDICTIONS_PREFIX}"
        )
    except Exception as e:
        logger.warning(f"No se pudo configurar cliente S3: {e}", exc_info=True)
        logger.warning("Las predicciones no se guardarán sin configuración S3")
else:
    logger.debug("S3_BUCKET no configurado, omitiendo configuración de S3")

# Log final de inicialización
logger.info("=" * 60)
logger.info("Aplicación Flask inicializada correctamente")
logger.info(
    f"Modelos cargados - Clasificación: {model_clasificacion is not None}, Segmentación: {model_segmentacion is not None}"
)
logger.info(f"S3 configurado: {s3_client is not None}, Storage: {storage is not None}")
logger.info("=" * 60)

# Preparación de imagen


def prepare_image(image_path_or_bytes, target):
    """Prepara una imagen para el modelo"""
    try:
        if isinstance(image_path_or_bytes, str):
            img = cv2.imread(image_path_or_bytes)
            if img is None:
                raise ValueError(
                    f"No se pudo cargar la imagen desde {image_path_or_bytes}"
                )
        else:
            img_array = np.frombuffer(image_path_or_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("No se pudo decodificar la imagen desde bytes")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = cv2.resize(img, target)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preparando imagen: {e}")
        raise


def mask_to_base64(mask):
    mask_normalized = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_normalized)
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ENDPOINTS


@app.route("/")
def home():
    """Endpoint raíz con información de la API"""
    return jsonify(
        {
            "status": "online",
            "service": "Brain Tumor Detection API",
            "models": {
                "clasificacion": model_clasificacion is not None,
                "segmentacion": model_segmentacion is not None,
            },
            "endpoints": {
                "health": "GET /health - Health check",
                "clasificacion": {
                    "predict_post": "POST /clasificacion/predict (Body: image)",
                    "predict_get": "GET /clasificacion/predict?id=<id>",
                    "predict_random": "GET /clasificacion/predict/random",
                },
                "segmentacion": {
                    "predict_post": "POST /segmentacion/predict (Body: image)",
                    "predict_get": "GET /segmentacion/predict?id=<id>",
                    "predict_random": "GET /segmentacion/predict/random",
                },
                "history": "GET /history (Query: type, limit)",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint para AWS App Runner"""
    try:
        models_loaded = (
            model_clasificacion is not None and model_segmentacion is not None
        )
        s3_configured = s3_client is not None
        storage_configured = storage is not None

        status = "healthy" if models_loaded else "degraded"

        return jsonify(
            {
                "status": status,
                "models_loaded": models_loaded,
                "s3_configured": s3_configured,
                "storage_configured": storage_configured,
                "timestamp": datetime.now().isoformat(),
            }
        ), 200 if models_loaded else 503
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# CLASIFICACIÓN


@app.route("/clasificacion/predict", methods=["GET", "POST"])
def clasificacion_predict():
    if request.method == "GET":
        prediction_id = request.args.get("id")
        if prediction_id:
            if not storage:
                return jsonify({"error": "Almacenamiento S3 no configurado"}), 503

            prediction = storage.get_prediction(prediction_id, "clasificacion")
            if prediction:
                return jsonify(
                    {
                        "id": prediction["id"],
                        "type": prediction["type"],
                        "date": prediction["date"],
                        "filename": prediction["filename"],
                        "predicted_class": prediction.get("predicted_class"),
                        "confidence": f"{prediction.get('confidence', 0):.2%}",
                    }
                )
            return jsonify({"error": "Predicción no encontrada"}), 404
        return jsonify(
            {
                "endpoint": "/clasificacion/predict",
                "methods": {
                    "POST": "Enviar imagen para predicción",
                    "GET": "Añadir ?id=<id> para consultar predicción",
                },
            }
        )

    data = {"success": False}
    image_bytes = None
    filename = "upload"

    try:
        if request.files.get("image"):
            image_file = request.files["image"]
            filename = image_file.filename or "upload"
            image_bytes = image_file.read()

            # Validar que la imagen no esté vacía
            if len(image_bytes) == 0:
                return jsonify({"error": "La imagen está vacía"}), 400

        if not image_bytes:
            return jsonify({"error": "No se proporcionó imagen"}), 400

        if not model_clasificacion:
            return jsonify({"error": "Modelo de clasificación no cargado"}), 503

        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        preds = model_clasificacion.predict(processed_image, verbose=0)
        pred_idx = np.argmax(preds, axis=1)[0]
        prob = float(np.max(preds))
        pred_label = LABELS[pred_idx]

        # Guardar en S3 si está configurado
        prediction_id = None
        if storage:
            try:
                prediction_id = storage.save_prediction(
                    prediction_type="clasificacion",
                    filename=filename,
                    predicted_class=pred_label,
                    confidence=prob,
                )
                logger.info(f"Predicción guardada en S3 con ID: {prediction_id}")
            except Exception as e:
                logger.error(f"Error guardando predicción en S3: {e}")
                # Continuar sin guardar si falla S3

        data.update(
            {
                "prediction_id": prediction_id,
                "prediction_label": pred_label,
                "confidence": f"{prob:.2%}",
                "success": True,
            }
        )
        logger.info(f"Predicción de clasificación exitosa: {pred_label} ({prob:.2%})")
        return jsonify(data)

    except ValueError as e:
        logger.error(f"Error de validación en clasificacion_predict: {e}")
        return jsonify({"error": f"Error procesando imagen: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error en clasificacion_predict: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500


@app.route("/clasificacion/predict/random", methods=["GET"])
def clasificacion_predict_random():
    """Predicción aleatoria desde S3 o local"""
    try:
        image_files = []

        # Intentar obtener imágenes desde S3
        if s3_client and S3_BUCKET:
            try:
                prefix = f"{S3_DATA_PREFIX.rstrip('/')}/"
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                if "Contents" in response:
                    image_files = [
                        obj["Key"]
                        for obj in response["Contents"]
                        if obj["Key"].endswith(".tif") and "_mask" not in obj["Key"]
                    ]
                    # Descargar imagen seleccionada
                    if image_files:
                        selected_key = random.choice(image_files)
                        temp_file = io.BytesIO()
                        s3_client.download_fileobj(S3_BUCKET, selected_key, temp_file)
                        temp_file.seek(0)
                        image_bytes = temp_file.read()
                        filename = os.path.basename(selected_key)
                    else:
                        return jsonify(
                            {"error": "No se encontraron imágenes en S3"}
                        ), 404
                else:
                    return jsonify({"error": "No se encontraron imágenes en S3"}), 404
            except Exception as e:
                logger.warning(
                    f"Error obteniendo imágenes desde S3: {e}, intentando local..."
                )
                image_files = []

        # Fallback: buscar imágenes locales
        if not image_files:
            # No hay carpeta local hardcodeada, retornar error
            return jsonify(
                {
                    "error": "No hay imágenes disponibles. Configure S3_BUCKET o proporcione una imagen."
                }
            ), 404

        if not image_bytes:
            return jsonify({"error": "No se pudo obtener la imagen"}), 404

        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        preds = model_clasificacion.predict(processed_image)
        pred_idx = np.argmax(preds, axis=1)[0]
        prob = float(np.max(preds))
        pred_label = LABELS[pred_idx]

        return jsonify(
            {
                "filename": filename,
                "prediction_label": pred_label,
                "confidence": f"{prob:.2%}",
            }
        )
    except Exception as e:
        logger.error(f"Error en clasificacion_predict_random: {e}")
        return jsonify({"error": str(e)}), 500


# SEGMENTACIÓN


@app.route("/segmentacion/predict", methods=["GET", "POST"])
def segmentacion_predict():
    if request.method == "GET":
        prediction_id = request.args.get("id")
        if prediction_id:
            if not storage:
                return jsonify({"error": "Almacenamiento S3 no configurado"}), 503

            prediction = storage.get_prediction(prediction_id, "segmentacion")
            if prediction:
                mask_b64 = prediction.get("mask_base64", "")
                return jsonify(
                    {
                        "id": prediction["id"],
                        "type": prediction["type"],
                        "date": prediction["date"],
                        "filename": prediction["filename"],
                        "mask_base64": f"data:image/png;base64,{mask_b64}"
                        if mask_b64
                        else None,
                    }
                )
            return jsonify({"error": "Segmentación no encontrada"}), 404
        return jsonify(
            {
                "endpoint": "/segmentacion/predict",
                "methods": {
                    "POST": "Enviar imagen para segmentación",
                    "GET": "Añadir ?id=<id> para consultar segmentación",
                },
            }
        )

    data = {"success": False}
    image_bytes = None
    filename = "upload"

    try:
        if request.files.get("image"):
            image_file = request.files["image"]
            filename = image_file.filename or "upload"
            image_bytes = image_file.read()

            # Validar que la imagen no esté vacía
            if len(image_bytes) == 0:
                return jsonify({"error": "La imagen está vacía"}), 400

        if not image_bytes:
            return jsonify({"error": "No se proporcionó imagen"}), 400

        if not model_segmentacion:
            return jsonify({"error": "Modelo de segmentación no cargado"}), 503

        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        mask = model_segmentacion.predict(processed_image, verbose=0)
        mask = mask[0, :, :, 0]
        mask_b64 = mask_to_base64(mask)

        # Guardar en S3 si está configurado
        segmentation_id = None
        if storage:
            try:
                segmentation_id = storage.save_prediction(
                    prediction_type="segmentacion",
                    filename=filename,
                    mask_base64=mask_b64,
                )
                logger.info(f"Segmentación guardada en S3 con ID: {segmentation_id}")
            except Exception as e:
                logger.error(f"Error guardando segmentación en S3: {e}")
                # Continuar sin guardar si falla S3

        data.update(
            {
                "segmentation_id": segmentation_id,
                "filename": filename,
                "mask_base64": f"data:image/png;base64,{mask_b64}",
                "success": True,
            }
        )
        logger.info(f"Segmentación exitosa para archivo: {filename}")
        return jsonify(data)

    except ValueError as e:
        logger.error(f"Error de validación en segmentacion_predict: {e}")
        return jsonify({"error": f"Error procesando imagen: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error en segmentacion_predict: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500


@app.route("/segmentacion/predict/random", methods=["GET"])
def segmentacion_predict_random():
    """Segmentación aleatoria desde S3 o local"""
    try:
        image_bytes = None
        filename = None

        # Intentar obtener imágenes desde S3
        if s3_client and S3_BUCKET:
            try:
                prefix = f"{S3_DATA_PREFIX.rstrip('/')}/"
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                if "Contents" in response:
                    image_files = [
                        obj["Key"]
                        for obj in response["Contents"]
                        if obj["Key"].endswith(".tif") and "_mask" not in obj["Key"]
                    ]
                    if image_files:
                        selected_key = random.choice(image_files)
                        temp_file = io.BytesIO()
                        s3_client.download_fileobj(S3_BUCKET, selected_key, temp_file)
                        temp_file.seek(0)
                        image_bytes = temp_file.read()
                        filename = os.path.basename(selected_key)
            except Exception as e:
                logger.warning(f"Error obteniendo imágenes desde S3: {e}")

        if not image_bytes:
            return jsonify(
                {
                    "error": "No hay imágenes disponibles. Configure S3_BUCKET o proporcione una imagen."
                }
            ), 404

        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        mask = model_segmentacion.predict(processed_image)
        mask = mask[0, :, :, 0]
        mask_b64 = mask_to_base64(mask)

        return jsonify(
            {"filename": filename, "mask_base64": f"data:image/png;base64,{mask_b64}"}
        )
    except Exception as e:
        logger.error(f"Error en segmentacion_predict_random: {e}")
        return jsonify({"error": str(e)}), 500


# HISTORY (desde S3)


@app.route("/history", methods=["GET"])
def get_history():
    if not storage:
        return jsonify({"error": "Almacenamiento S3 no configurado"}), 503

    try:
        limit = int(request.args.get("limit", 10))
        type_filter = request.args.get("type")  # "clasificacion" o "segmentacion"

        history = storage.get_history(limit=limit, prediction_type=type_filter)

        # Formatear respuesta
        formatted_history = []
        for pred in history:
            item = {
                "id": pred["id"],
                "type": pred["type"],
                "date": pred["date"],
                "filename": pred["filename"],
            }
            if pred["type"] == "clasificacion":
                item["predicted_class"] = pred.get("predicted_class")
                confidence = pred.get("confidence", 0)
                item["confidence"] = (
                    f"{confidence:.2%}"
                    if isinstance(confidence, (int, float))
                    else confidence
                )
            else:
                mask_b64 = pred.get("mask_base64", "")
                item["mask_base64"] = (
                    f"data:image/png;base64,{mask_b64}" if mask_b64 else None
                )
            formatted_history.append(item)

        return jsonify({"count": len(formatted_history), "data": formatted_history})
    except ValueError as e:
        return jsonify({"error": f"Parámetro limit inválido: {e}"}), 400
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"Iniciando servidor Flask en {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
