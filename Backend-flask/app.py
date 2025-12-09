import numpy as np
import os
import base64
import sqlite3
import random
import glob
from datetime import datetime
from flask import Flask, request, jsonify, g
from tensorflow.keras.models import load_model
from PIL import Image
import io
import cv2
import tensorflow as tf

# Inicializamos la aplicación Flask
app = Flask(__name__)
application = app

# Funciones para la segmentación
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl

def iou_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

# Bases
TARGET_SIZE = (256, 256)
IMAGES_FOLDER = '../Mini_base_datos'
LABELS = ["No detectado (0)", "Detectado(1)"]
DB_FILE = "hospital_data.db"

# Modelos
MODEL_CLASIFICACION = '../models/classifier-resnet-model-final.keras'
MODEL_SEGMENTACION = '../models/segmentation_ResUNet_final.keras'

model_clasificacion = None
model_segmentacion = None

try:
    if os.path.exists(MODEL_CLASIFICACION):
        print(f"Cargando modelo clasificación desde {MODEL_CLASIFICACION}...")
        model_clasificacion = load_model(MODEL_CLASIFICACION)
        print("¡Modelo clasificación cargado!")
    else:
        print(f"ADVERTENCIA: No se encontró {MODEL_CLASIFICACION}")
except Exception as e:
    print(f"Error cargando modelo clasificación: {e}")

try:
    if os.path.exists(MODEL_SEGMENTACION):
        print(f"Cargando modelo segmentación desde {MODEL_SEGMENTACION}...")
        model_segmentacion = load_model(MODEL_SEGMENTACION, custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou_coef': iou_coef
        })
        print("¡Modelo segmentación cargado!")
    else:
        print(f"ADVERTENCIA: No se encontró {MODEL_SEGMENTACION}")
except Exception as e:
    print(f"Error cargando modelo segmentación: {e}")

# Base de datos

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_FILE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                date TEXT,
                filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                mask_base64 TEXT
            )
        ''')
        db.commit()

init_db()

# Preparación de imagen

def prepare_image(image_path_or_bytes, target):
    if isinstance(image_path_or_bytes, str):
        img = cv2.imread(image_path_or_bytes)
    else:
        img_array = np.frombuffer(image_path_or_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = cv2.resize(img, target)
    img = np.expand_dims(img, axis=0)
    return img

def mask_to_base64(mask):
    mask_normalized = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_normalized)
    buffer = io.BytesIO()
    mask_img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')





# ENDPOINTS


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "models": {
            "clasificacion": model_clasificacion is not None,
            "segmentacion": model_segmentacion is not None
        },
        "endpoints": {
            "clasificacion": {
                "predict_post": "POST /clasificacion/predict (Body: image)",
                "predict_get": "GET /clasificacion/predict?id=<id>",
                "predict_random": "GET /clasificacion/predict/random"
            },
            "segmentacion": {
                "predict_post": "POST /segmentacion/predict (Body: image)",
                "predict_get": "GET /segmentacion/predict?id=<id>",
                "predict_random": "GET /segmentacion/predict/random"
            },
            "history": "GET /history (Query: type, limit)"
        }
    })


# CLASIFICACIÓN


@app.route("/clasificacion/predict", methods=["GET", "POST"])
def clasificacion_predict():
    if request.method == "GET":
        prediction_id = request.args.get('id')
        if prediction_id:
            db = get_db()
            cursor = db.execute("SELECT * FROM predictions WHERE id = ? AND type = 'clasificacion'", (prediction_id,))
            row = cursor.fetchone()
            if row:
                return jsonify({
                    "id": row["id"],
                    "type": row["type"],
                    "date": row["date"],
                    "filename": row["filename"],
                    "predicted_class": row["predicted_class"],
                    "confidence": f"{row['confidence']:.2%}"
                })
            return jsonify({"error": "Predicción no encontrada"}), 404
        return jsonify({
            "endpoint": "/clasificacion/predict",
            "methods": {
                "POST": "Enviar imagen para predicción",
                "GET": "Añadir ?id=<id> para consultar predicción"
            }
        })
    
    data = {"success": False}
    image_bytes = None
    filename = "upload"

    if request.files.get("image"):
        image_file = request.files["image"]
        filename = image_file.filename
        image_bytes = image_file.read()

    if image_bytes and model_clasificacion:
        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        preds = model_clasificacion.predict(processed_image)
        pred_idx = np.argmax(preds, axis=1)[0]
        prob = float(np.max(preds))
        pred_label = LABELS[pred_idx]

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO predictions (type, date, filename, predicted_class, confidence) VALUES (?, ?, ?, ?, ?)',
            ('clasificacion', datetime.now().isoformat(), filename, pred_label, prob)
        )
        db.commit()
        prediction_id = cursor.lastrowid

        data.update({
            "prediction_id": prediction_id,
            "prediction_label": pred_label,
            "confidence": f"{prob:.2%}",
            "success": True
        })
        return jsonify(data)

    return jsonify({"error": "Falta imagen o modelo no cargado"}), 400


@app.route("/clasificacion/predict/random", methods=["GET"])
def clasificacion_predict_random():
    all_files = glob.glob(os.path.join(IMAGES_FOLDER, '*.tif'))
    image_files = [f for f in all_files if '_mask' not in f]
    
    if not image_files:
        return jsonify({"error": "No se encontraron imágenes en la carpeta"}), 404
    
    selected_image = random.choice(image_files)
    filename = os.path.basename(selected_image)
    
    with open(selected_image, 'rb') as f:
        image_bytes = f.read()
    
    processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
    preds = model_clasificacion.predict(processed_image)
    pred_idx = np.argmax(preds, axis=1)[0]
    prob = float(np.max(preds))
    pred_label = LABELS[pred_idx]
    
    return jsonify({
        "filename": filename,
        "prediction_label": pred_label,
        "confidence": f"{prob:.2%}"
    })


# SEGMENTACIÓN
# #  pred_mask = (pred_mask > 0.5).astype(np.uint8)
#   plt.imshow(pred_mask, cmap='gray')
#   plt.axis('off')

@app.route("/segmentacion/predict", methods=["GET", "POST"])
def segmentacion_predict():
    if request.method == "GET":
        prediction_id = request.args.get('id')
        if prediction_id:
            db = get_db()
            cursor = db.execute("SELECT * FROM predictions WHERE id = ? AND type = 'segmentacion'", (prediction_id,))
            row = cursor.fetchone()
            if row:
                return jsonify({
                    "id": row["id"],
                    "type": row["type"],
                    "date": row["date"],
                    "filename": row["filename"],
                    "mask_base64": f"data:image/png;base64,{row['mask_base64']}"
                })
            return jsonify({"error": "Segmentación no encontrada"}), 404
        return jsonify({
            "endpoint": "/segmentacion/predict",
            "methods": {
                "POST": "Enviar imagen para segmentación",
                "GET": "Añadir ?id=<id> para consultar segmentación"
            }
        })
    
    data = {"success": False}
    image_bytes = None
    filename = "upload"

    if request.files.get("image"):
        image_file = request.files["image"]
        filename = image_file.filename
        image_bytes = image_file.read()

    if image_bytes and model_segmentacion:
        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        mask = model_segmentacion.predict(processed_image)
        mask = mask[0, :, :, 0]
        mask_b64 = mask_to_base64(mask)

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO predictions (type, date, filename, mask_base64) VALUES (?, ?, ?, ?)',
            ('segmentacion', datetime.now().isoformat(), filename, mask_b64)
        )
        db.commit()
        segmentation_id = cursor.lastrowid

        data.update({
            "segmentation_id": segmentation_id,
            "filename": filename,
            "mask_base64": f"data:image/png;base64,{mask_b64}",
            "success": True
        })
        return jsonify(data)

    return jsonify({"error": "Falta imagen o modelo no cargado"}), 400


@app.route("/segmentacion/predict/random", methods=["GET"])
def segmentacion_predict_random():
    all_files = glob.glob(os.path.join(IMAGES_FOLDER, '*.tif'))
    image_files = [f for f in all_files if '_mask' not in f]
    
    if not image_files:
        return jsonify({"error": "No se encontraron imágenes en la carpeta"}), 404
    
    selected_image = random.choice(image_files)
    filename = os.path.basename(selected_image)
    
    with open(selected_image, 'rb') as f:
        image_bytes = f.read()
    
    processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
    mask = model_segmentacion.predict(processed_image)
    mask = mask[0, :, :, 0]
    mask_b64 = mask_to_base64(mask)
    
    return jsonify({
        "filename": filename,
        "mask_base64": f"data:image/png;base64,{mask_b64}"
    })


# HISTORY (base de datos)


@app.route("/history", methods=["GET"])
def get_history():
    db = get_db()
    limit = request.args.get('limit', 10)
    type_filter = request.args.get('type')

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
            "filename": row["filename"]
        }
        if row["type"] == "clasificacion":
            item["predicted_class"] = row["predicted_class"]
            item["confidence"] = f"{row['confidence']:.2%}"
        else:
            item["mask_base64"] = f"data:image/png;base64,{row['mask_base64']}" if row["mask_base64"] else None
        history.append(item)

    return jsonify({"count": len(history), "data": history})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)