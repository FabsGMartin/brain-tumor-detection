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

# --- FUNCIONES DE PÉRDIDA PERSONALIZADAS ---
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

# --- CONFIGURACIÓN ---
MODEL_VERSION = "1.0.0"
TARGET_SIZE = (256, 256)
IMAGES_FOLDER = '../Mini_base_datos'
DB_FILE = "hospital_data_segmentation.db"

# --- CARGA DEL MODELO (.keras) ---
MODEL_FILE = '../models/segmentation_ResUNet4.keras'
model = None

try:
    if os.path.exists(MODEL_FILE):
        print(f"Cargando modelo desde {MODEL_FILE}...")
        model = load_model(MODEL_FILE, custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou_coef': iou_coef
        })
        print("¡Modelo .keras cargado exitosamente!")
    else:
        print(f"ADVERTENCIA: No se encontró {MODEL_FILE}. La segmentación fallará.")
except Exception as e:
    print(f"Error cargando modelo: {e}")

# --- GESTIÓN DE BASE DE DATOS (SQLite) ---

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
            CREATE TABLE IF NOT EXISTS segmentations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                filename TEXT,
                mask_base64 TEXT
            )
        ''')
        db.commit()

init_db()

# --- FUNCIONES AUXILIARES DE IMAGEN ---

def prepare_image(image_path_or_bytes, target):
    """Preprocesa la imagen igual que en entrenamiento: BGR a RGB, normalizado /255"""
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
    """Convierte la máscara (array numpy) a base64 PNG"""
    mask_normalized = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_normalized)
    buffer = io.BytesIO()
    mask_img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# ==========================================
# DEFINICIÓN DE ENDPOINTS
# ==========================================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "model_type": ".keras - ResUNet Segmentation",
        "endpoints": {
            "predict_post": "POST /predict (Body: image) - Enviar imagen para segmentación",
            "predict_get": "GET /predict/<id> - Obtener segmentación por ID",
            "predict_random": "GET /predict/random - Segmentación con imagen aleatoria",
            "history": "GET /history (Query: limit) - Historial de segmentaciones"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    image_bytes = None
    filename = "upload"

    if request.files.get("image"):
        image_file = request.files["image"]
        filename = image_file.filename
        image_bytes = image_file.read()

    if image_bytes and model:
        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        mask = model.predict(processed_image)
        mask = mask[0, :, :, 0]
        mask_b64 = mask_to_base64(mask)

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO segmentations (date, filename, mask_base64) VALUES (?, ?, ?)',
            (datetime.now().isoformat(), filename, mask_b64)
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

@app.route("/predict/<int:segmentation_id>", methods=["GET"])
def get_segmentation(segmentation_id):
    db = get_db()
    cursor = db.execute("SELECT * FROM segmentations WHERE id = ?", (segmentation_id,))
    row = cursor.fetchone()
    
    if not row:
        return jsonify({"error": "Segmentación no encontrada"}), 404
    
    return jsonify({
        "id": row["id"],
        "date": row["date"],
        "filename": row["filename"],
        "mask_base64": f"data:image/png;base64,{row['mask_base64']}"
    })

@app.route("/predict/random", methods=["GET"])
def predict_random():
    all_files = glob.glob(os.path.join(IMAGES_FOLDER, '*.tif'))
    image_files = [f for f in all_files if '_mask' not in f]
    
    if not image_files:
        return jsonify({"error": "No se encontraron imágenes en la carpeta"}), 404
    
    selected_image = random.choice(image_files)
    filename = os.path.basename(selected_image)
    
    with open(selected_image, 'rb') as f:
        image_bytes = f.read()
    
    processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
    mask = model.predict(processed_image)
    mask = mask[0, :, :, 0]
    mask_b64 = mask_to_base64(mask)
    
    return jsonify({
        "filename": filename,
        "mask_base64": f"data:image/png;base64,{mask_b64}"
    })

@app.route("/history", methods=["GET"])
def get_history():
    db = get_db()
    limit = request.args.get('limit', 10)
    cursor = db.execute("SELECT id, date, filename FROM segmentations ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    return jsonify({"count": len(history), "data": history})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)