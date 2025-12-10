import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model


# Funciones para la segmentación
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
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


# Custom objects para cargar modelos
CUSTOM_OBJECTS = {
    "bce_dice_loss": bce_dice_loss,
    "dice_coef": dice_coef,
    "dice_loss": dice_loss,
    "iou_coef": iou_coef,
}

# ---------- PATH CONSTANTS ----------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_CLASIFICACION = MODELS_DIR / "classifier-resnet-model-final.keras"
MODEL_SEGMENTACION = MODELS_DIR / "segmentation_ResUNet_final.keras"

model_clasificacion = None
model_segmentacion = None


def load_models():
    """Carga los modelos de clasificación y segmentación"""
    global model_clasificacion, model_segmentacion

    # Obtener las rutas completas a los modelos
    model_clas_path = MODEL_CLASIFICACION
    model_seg_path = MODEL_SEGMENTACION

    # Cargar modelo de clasificación
    try:
        if model_clas_path.exists():
            print(f"Cargando modelo clasificación desde {model_clas_path}...")
            model_clasificacion = load_model(str(model_clas_path))
            print("¡Modelo clasificación cargado!")
        else:
            print(f"ADVERTENCIA: No se encontró {model_clas_path}")
    except Exception as e:
        print(f"Error cargando modelo clasificación: {e}")

    # Cargar modelo de segmentación
    try:
        if model_seg_path.exists():
            print(f"Cargando modelo segmentación desde {model_seg_path}...")
            model_segmentacion = load_model(
                str(model_seg_path), custom_objects=CUSTOM_OBJECTS
            )
            print("¡Modelo segmentación cargado!")
        else:
            print(f"ADVERTENCIA: No se encontró {model_seg_path}")
    except Exception as e:
        print(f"Error cargando modelo segmentación: {e}")

    return model_clasificacion, model_segmentacion


# Cargar modelos al importar el módulo
load_models()
