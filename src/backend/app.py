import numpy as np
import os
import base64
import random
import glob
import logging
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

# Importar módulos locales
from model import model_clasificacion, model_segmentacion
from database import get_db, init_db, close_connection

# Cargar variables de entorno
load_dotenv()

# Configurar logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializamos la aplicación Flask
app = Flask(__name__)
application = app

# Configurar CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
CORS(app, origins=cors_origins, supports_credentials=True)
logger.info(f"CORS configurado para origins: {cors_origins}")

# Registrar teardown handler
app.teardown_appcontext(close_connection)

# Inicializar base de datos
init_db(app)

# Configuración S3
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET")
S3_DATA_PREFIX = os.getenv("S3_DATA_PREFIX", "data/")

# ---------- PATH CONSTANTS ----------
BASE_DIR = Path(__file__).resolve().parent

# Constantes desde variables de entorno
target_size_str = os.getenv("TARGET_SIZE", "256,256")
TARGET_SIZE = tuple(map(int, target_size_str.split(",")))
LABELS = ["No detectado (0)", "Detectado(1)"]

# Configurar cliente S3 si está disponible
s3_client = None
if S3_DATA_BUCKET and os.getenv("AWS_ACCESS_KEY_ID"):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        logger.info(f"Cliente S3 configurado para bucket: {S3_DATA_BUCKET}")
    except Exception as e:
        logger.warning(f"No se pudo configurar cliente S3: {e}")

# Preparación de imagen


def prepare_image(image_path_or_bytes, target):
    """Prepara una imagen para el modelo"""
    try:
        if isinstance(image_path_or_bytes, str):
            img = cv2.imread(image_path_or_bytes)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen desde {image_path_or_bytes}")
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
        models_loaded = model_clasificacion is not None and model_segmentacion is not None
        s3_configured = s3_client is not None
        
        status = "healthy" if models_loaded else "degraded"
        
        return jsonify({
            "status": status,
            "models_loaded": models_loaded,
            "s3_configured": s3_configured,
            "timestamp": datetime.now().isoformat()
        }), 200 if models_loaded else 503
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# CLASIFICACIÓN


@app.route("/clasificacion/predict", methods=["GET", "POST"])
def clasificacion_predict():
    if request.method == "GET":
        prediction_id = request.args.get("id")
        if prediction_id:
            db = get_db()
            cursor = db.execute(
                "SELECT * FROM predictions WHERE id = ? AND type = 'clasificacion'",
                (prediction_id,),
            )
            row = cursor.fetchone()
            if row:
                return jsonify(
                    {
                        "id": row["id"],
                        "type": row["type"],
                        "date": row["date"],
                        "filename": row["filename"],
                        "predicted_class": row["predicted_class"],
                        "confidence": f"{row['confidence']:.2%}",
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

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO predictions (type, date, filename, predicted_class, confidence) VALUES (?, ?, ?, ?, ?)",
            ("clasificacion", datetime.now().isoformat(), filename, pred_label, prob),
        )
        db.commit()
        prediction_id = cursor.lastrowid

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
        if s3_client and S3_DATA_BUCKET:
            try:
                prefix = f"{S3_DATA_PREFIX.rstrip('/')}/"
                response = s3_client.list_objects_v2(
                    Bucket=S3_DATA_BUCKET,
                    Prefix=prefix
                )
                if 'Contents' in response:
                    image_files = [
                        obj['Key'] for obj in response['Contents']
                        if obj['Key'].endswith('.tif') and '_mask' not in obj['Key']
                    ]
                    # Descargar imagen seleccionada
                    if image_files:
                        selected_key = random.choice(image_files)
                        temp_file = io.BytesIO()
                        s3_client.download_fileobj(S3_DATA_BUCKET, selected_key, temp_file)
                        temp_file.seek(0)
                        image_bytes = temp_file.read()
                        filename = os.path.basename(selected_key)
                    else:
                        return jsonify({"error": "No se encontraron imágenes en S3"}), 404
                else:
                    return jsonify({"error": "No se encontraron imágenes en S3"}), 404
            except Exception as e:
                logger.warning(f"Error obteniendo imágenes desde S3: {e}, intentando local...")
                image_files = []
        
        # Fallback: buscar imágenes locales
        if not image_files:
            # No hay carpeta local hardcodeada, retornar error
            return jsonify({
                "error": "No hay imágenes disponibles. Configure S3_DATA_BUCKET o proporcione una imagen."
            }), 404

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
            db = get_db()
            cursor = db.execute(
                "SELECT * FROM predictions WHERE id = ? AND type = 'segmentacion'",
                (prediction_id,),
            )
            row = cursor.fetchone()
            if row:
                return jsonify(
                    {
                        "id": row["id"],
                        "type": row["type"],
                        "date": row["date"],
                        "filename": row["filename"],
                        "mask_base64": f"data:image/png;base64,{row['mask_base64']}",
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

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO predictions (type, date, filename, mask_base64) VALUES (?, ?, ?, ?)",
            ("segmentacion", datetime.now().isoformat(), filename, mask_b64),
        )
        db.commit()
        segmentation_id = cursor.lastrowid

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
        if s3_client and S3_DATA_BUCKET:
            try:
                prefix = f"{S3_DATA_PREFIX.rstrip('/')}/"
                response = s3_client.list_objects_v2(
                    Bucket=S3_DATA_BUCKET,
                    Prefix=prefix
                )
                if 'Contents' in response:
                    image_files = [
                        obj['Key'] for obj in response['Contents']
                        if obj['Key'].endswith('.tif') and '_mask' not in obj['Key']
                    ]
                    if image_files:
                        selected_key = random.choice(image_files)
                        temp_file = io.BytesIO()
                        s3_client.download_fileobj(S3_DATA_BUCKET, selected_key, temp_file)
                        temp_file.seek(0)
                        image_bytes = temp_file.read()
                        filename = os.path.basename(selected_key)
            except Exception as e:
                logger.warning(f"Error obteniendo imágenes desde S3: {e}")
        
        if not image_bytes:
            return jsonify({
                "error": "No hay imágenes disponibles. Configure S3_DATA_BUCKET o proporcione una imagen."
            }), 404

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


# HISTORY (base de datos)


@app.route("/history", methods=["GET"])
def get_history():
    db = get_db()
    limit = request.args.get("limit", 10)
    type_filter = request.args.get("type")

    query = "SELECT id, type, date, filename, predicted_class, confidence, mask_base64 FROM predictions"
    params = []

    if type_filter:
        query += " WHERE type = ?"
        params.append(type_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor = db.execute(query, params)
    rows = cursor.fetchall()

    history = []
    for row in rows:
        item = {
            "id": row["id"],
            "type": row["type"],
            "date": row["date"],
            "filename": row["filename"],
        }
        if row["type"] == "clasificacion":
            item["predicted_class"] = row["predicted_class"]
            item["confidence"] = f"{row['confidence']:.2%}"
        else:
            item["mask_base64"] = (
                f"data:image/png;base64,{row['mask_base64']}"
                if row["mask_base64"]
                else None
            )
        history.append(item)

    return jsonify({"count": len(history), "data": history})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Iniciando servidor Flask en {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
