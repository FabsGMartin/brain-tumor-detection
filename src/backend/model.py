import os
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


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
MODELS_DIR.mkdir(exist_ok=True)  # Crear directorio si no existe

# Configuración desde variables de entorno
MODEL_CLASSIFICATION_NAME = os.getenv(
    "MODEL_CLASSIFICATION_NAME", "classifier-resnet-model-final.keras"
)
MODEL_SEGMENTATION_NAME = os.getenv(
    "MODEL_SEGMENTATION_NAME", "segmentation_ResUNet_final.keras"
)

# Rutas locales
MODEL_CLASIFICACION_LOCAL = MODELS_DIR / MODEL_CLASSIFICATION_NAME
MODEL_SEGMENTACION_LOCAL = MODELS_DIR / MODEL_SEGMENTATION_NAME

model_clasificacion = None
model_segmentacion = None


def load_model_local(model_name, custom_objects=None):
    """Carga un modelo desde archivo local"""
    local_path = MODELS_DIR / model_name
    if local_path.exists():
        try:
            logger.info(f"Cargando modelo {model_name} desde archivo local: {local_path}")
            if custom_objects:
                model = load_model(str(local_path), custom_objects=custom_objects)
            else:
                model = load_model(str(local_path))
            logger.info(f"¡Modelo {model_name} cargado exitosamente desde archivo local!")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo desde archivo local: {e}")
            return None
    else:
        logger.error(
            f"Modelo {model_name} no encontrado localmente en {local_path}"
        )
        return None


def load_models():
    """Carga los modelos de clasificación y segmentación desde archivos locales"""
    global model_clasificacion, model_segmentacion

    # Cargar modelo de clasificación
    model_clasificacion = load_model_local(MODEL_CLASSIFICATION_NAME)

    # Cargar modelo de segmentación
    model_segmentacion = load_model_local(
        MODEL_SEGMENTATION_NAME,
        custom_objects=CUSTOM_OBJECTS,
    )

    return model_clasificacion, model_segmentacion


# Cargar modelos al importar el módulo
load_models()
