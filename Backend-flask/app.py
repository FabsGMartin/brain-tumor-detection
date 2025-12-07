import numpy as np
import os
import base64
import re
import sqlite3
import time
from datetime import datetime
from flask import Flask, request, jsonify, g
from tensorflow.keras.models import load_model
from PIL import Image
import io
import cv2
import random
import glob

# Inicializamos la aplicación Flask
app = Flask(__name__)
application = app

# --- CONFIGURACIÓN ---
MODEL_VERSION = "1.0.0"
TARGET_SIZE = (256, 256)
IMAGES_FOLDER = '../Mini_base_datos'
LABELS = ["No detectado (0)", "Detectado(1)"]
DB_FILE = "hospital_data.db"

# --- CARGA DEL MODELO (.keras) ---
MODEL_FILE = '../models/classifier-resnet-model9.keras'
model = None

try:
    if os.path.exists(MODEL_FILE):
        print(f"Cargando modelo desde {MODEL_FILE}...")
        model = load_model(MODEL_FILE)
        print("¡Modelo .keras cargado exitosamente!")
    else:
        print(f"ADVERTENCIA: No se encontró {MODEL_FILE}. La predicción fallará.")
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
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                corrected_label TEXT
            )
        ''')
        db.commit()

init_db()

# --- FUNCIONES AUXILIARES DE IMAGEN ---
def decode_base64_image(base64_string):
    image_data = re.sub('^data:image/.+;base64,', '', base64_string)
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

def prepare_image(image_path_or_bytes, target):
    """Preprocesa la imagen igual que en entrenamiento: BGR, normalizado /255"""
    if isinstance(image_path_or_bytes, str):
        # Si es ruta de archivo
        img = cv2.imread(image_path_or_bytes)
    else:
        # Si son bytes (de la request)
        img_array = np.frombuffer(image_path_or_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # BGR a RGB y normalizar (como en entrenamiento)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = cv2.resize(img, target)
    img = np.expand_dims(img, axis=0)
    return img

# ==========================================
# DEFINICIÓN DE ENDPOINTS
# ==========================================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "model_type": ".keras (Modern Format)",
        "endpoints": {
            "predict_post": "POST /predict (Body: image) - Enviar imagen para predicción",
            "predict_get": "GET /predict/<id> - Obtener predicción por ID",
            "predict_random": "GET /predict/random - Predicción con imagen aleatoria",
            "predict_random_view": "GET /predict/random/view - Ver predicción en navegador",
            "history": "GET /history (Query: limit, class) - Historial de predicciones",
            "feedback": "PUT /feedback/<id> (Path + Body) - Corregir predicción",
            "train": "POST /admin/train (Body: epochs, lr) - Simular entrenamiento"
        }
    })



@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    image_bytes = None
    filename = "upload_base64"

    print("=== DEBUG ===")
    print("Files recibidos:", request.files)
    print("Modelo cargado:", model is not None)

    if request.files.get("image"):
        image_file = request.files["image"]
        filename = image_file.filename
        image_bytes = image_file.read()  # Guardamos bytes directamente
    elif request.json and "image" in request.json:
        image_bytes = decode_base64_image(request.json["image"])

    if image_bytes and model:
        processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
        preds = model.predict(processed_image)
        pred_idx = np.argmax(preds, axis=1)[0]
        prob = float(np.max(preds))
        pred_label = LABELS[pred_idx]

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO predictions (date, filename, predicted_class, confidence) VALUES (?, ?, ?, ?)',
            (datetime.now().isoformat(), filename, pred_label, prob)
        )
        db.commit()
        prediction_id = cursor.lastrowid

        data.update({
            "prediction_id": prediction_id,
            "prediction": pred_label,
            "confidence": f"{prob:.2%}",
            "success": True
        })
        return jsonify(data)

    return jsonify({"error": "Falta imagen o modelo no cargado"}), 400


@app.route("/predict/<int:prediction_id>", methods=["GET"])
def get_prediction(prediction_id):
    db = get_db()
    cursor = db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    
    if not row:
        return jsonify({"error": "Predicción no encontrada"}), 404
    
    return jsonify(dict(row))


@app.route("/predict/random", methods=["GET"])
def predict_random():
    # 1. Listar imágenes .tif (excluyendo máscaras)
    all_files = glob.glob(os.path.join(IMAGES_FOLDER, '*.tif'))
    image_files = [f for f in all_files if '_mask' not in f]
    
    if not image_files:
        return jsonify({"error": "No se encontraron imágenes en la carpeta"}), 404
    
    # 2. Elegir una aleatoria
    selected_image = random.choice(image_files)
    filename = os.path.basename(selected_image)
    
    # 3. Leer y procesar imagen
    with open(selected_image, 'rb') as f:
        image_bytes = f.read()
    
    processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
    
    # 4. Predecir
    preds = model.predict(processed_image)
    pred_idx = np.argmax(preds, axis=1)[0]
    prob = float(np.max(preds))
    pred_label = LABELS[pred_idx]
    
    return jsonify({
        "filename": filename,
        "prediction": pred_label,
        "confidence": f"{prob:.2%}"
    })


@app.route("/predict/random/view", methods=["GET"])
def predict_random_view():
    # 1. Listar imágenes .tif (excluyendo máscaras)
    all_files = glob.glob(os.path.join(IMAGES_FOLDER, '*.tif'))
    image_files = [f for f in all_files if '_mask' not in f]
    
    if not image_files:
        return "<h1>Error: No se encontraron imágenes</h1>", 404
    
    # 2. Elegir una aleatoria
    selected_image = random.choice(image_files)
    filename = os.path.basename(selected_image)
    
    # 3. Leer y procesar imagen
    with open(selected_image, 'rb') as f:
        image_bytes = f.read()
    
    processed_image = prepare_image(image_bytes, target=TARGET_SIZE)
    
    # 4. Predecir
    preds = model.predict(processed_image)
    pred_idx = np.argmax(preds, axis=1)[0]
    prob = float(np.max(preds))
    pred_label = LABELS[pred_idx]
    
    # 5. Convertir imagen a base64
    img = Image.open(io.BytesIO(image_bytes))
    png_buffer = io.BytesIO()
    img.save(png_buffer, format='PNG')
    png_buffer.seek(0)
    image_base64 = base64.b64encode(png_buffer.read()).decode('utf-8')
    
    # 6. Devolver HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Tumor Detection</title>
        <style>
            body {{ font-family: Arial; text-align: center; padding: 20px; background: #f0f0f0; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; max-width: 600px; margin: auto; }}
            img {{ max-width: 400px; border: 2px solid #333; border-radius: 5px; }}
            .prediction {{ font-size: 24px; margin: 20px 0; padding: 15px; border-radius: 5px; }}
            .detectado {{ background: #ffcccc; color: #cc0000; }}
            .no-detectado {{ background: #ccffcc; color: #006600; }}
            .confidence {{ font-size: 18px; color: #666; }}
            .filename {{ font-size: 14px; color: #999; }}
            .refresh {{ margin-top: 20px; padding: 10px 30px; font-size: 16px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Brain Tumor Detection</h1>
            <img src="data:image/png;base64,{image_base64}" alt="MRI Scan">
            <div class="prediction {'detectado' if pred_idx == 1 else 'no-detectado'}">
                {pred_label}
            </div>
            <div class="confidence">Confianza: {prob:.2%}</div>
            <div class="filename">Archivo: {filename}</div>
            <button class="refresh" onclick="location.reload()">🔄 Nueva imagen</button>
        </div>
    </body>
    </html>
    """
    return html




@app.route("/history", methods=["GET"])
def get_history():
    db = get_db()
    limit = request.args.get('limit', 10)
    class_filter = request.args.get('class_filter')

    query = "SELECT * FROM predictions"
    params = []

    if class_filter:
        query += " WHERE predicted_class = ?"
        params.append(class_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor = db.execute(query, params)
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]

    return jsonify({"count": len(history), "data": history})

@app.route("/feedback/<int:prediction_id>", methods=["PUT"])
def update_feedback(prediction_id):
    if not request.json or 'correct_label' not in request.json:
        return jsonify({"error": "Falta 'correct_label' en el JSON body"}), 400

    new_label = request.json['correct_label']

    if new_label not in LABELS:
        return jsonify({"error": f"Clase inválida. Use: {LABELS}"}), 400

    db = get_db()
    cursor = db.execute("SELECT id FROM predictions WHERE id = ?", (prediction_id,))
    if not cursor.fetchone():
        return jsonify({"error": "ID de predicción no encontrado"}), 404

    db.execute(
        "UPDATE predictions SET corrected_label = ? WHERE id = ?",
        (new_label, prediction_id)
    )
    db.commit()

    return jsonify({"message": f"Feedback guardado para predicción {prediction_id}", "status": "updated"})

@app.route("/admin/train", methods=["POST"])
def train_model():
    params = request.json or {}
    epochs = params.get("epochs", 5)
    learning_rate = params.get("learning_rate", 0.001)

    print(f"Iniciando entrenamiento: Epochs={epochs}, LR={learning_rate}")
    time.sleep(2)

    global MODEL_VERSION
    MODEL_VERSION = f"2.0.{int(time.time())}"

    return jsonify({
        "message": "Entrenamiento finalizado (Simulado)",
        "new_version": MODEL_VERSION,
        "config_used": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "architecture": "ResNet50 (.keras)"
        }
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)