import os
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import tempfile

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
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODELS_PREFIX = os.getenv("S3_MODELS_PREFIX", "models/")
MODEL_CLASSIFICATION_NAME = os.getenv(
    "MODEL_CLASSIFICATION_NAME", "classifier-resnet-model-final.keras"
)
MODEL_SEGMENTATION_NAME = os.getenv(
    "MODEL_SEGMENTATION_NAME", "segmentation_ResUNet_final.keras"
)

# Rutas locales (fallback)
MODEL_CLASIFICACION_LOCAL = MODELS_DIR / MODEL_CLASSIFICATION_NAME
MODEL_SEGMENTACION_LOCAL = MODELS_DIR / MODEL_SEGMENTATION_NAME

model_clasificacion = None
model_segmentacion = None


def download_from_s3(bucket_name, s3_key, local_path):
    """Descarga un archivo desde S3 a una ruta local"""
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
            logger.info(
                "Cliente S3 configurado con credenciales explícitas para descarga de modelos"
            )
        else:
            # Producción (App Runner): usar IAM role
            s3_client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-3"),
            )
            logger.info(
                "Cliente S3 configurado con IAM role (credenciales automáticas) para descarga de modelos"
            )

        logger.info(f"Descargando {s3_key} desde S3 bucket {bucket_name}...")
        s3_client.download_file(bucket_name, s3_key, str(local_path))
        logger.info(f"Modelo descargado exitosamente a {local_path}")
        return True
    except ClientError as e:
        logger.error(f"Error descargando desde S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado descargando desde S3: {e}")
        return False


def load_model_from_s3_or_local(model_name, model_key, custom_objects=None):
    """Carga un modelo desde S3 o desde archivo local (fallback)"""
    # Crear ruta temporal para descargar desde S3
    temp_model_path = MODELS_DIR / model_name

    # Intentar cargar desde S3 si está configurado
    if S3_BUCKET:
        s3_key = f"{S3_MODELS_PREFIX.rstrip('/')}/{model_name}"

        # Intentar descargar desde S3
        if download_from_s3(S3_BUCKET, s3_key, temp_model_path):
            try:
                logger.info(f"Cargando modelo {model_name} desde S3...")
                if custom_objects:
                    model = load_model(
                        str(temp_model_path), custom_objects=custom_objects
                    )
                else:
                    model = load_model(str(temp_model_path))
                logger.info(f"¡Modelo {model_name} cargado desde S3!")
                return model
            except Exception as e:
                logger.warning(f"Error cargando modelo desde S3, intentando local: {e}")

    # Fallback: cargar desde archivo local
    local_path = MODELS_DIR / model_name
    if local_path.exists():
        try:
            logger.info(f"Cargando modelo {model_name} desde archivo local...")
            if custom_objects:
                model = load_model(str(local_path), custom_objects=custom_objects)
            else:
                model = load_model(str(local_path))
            logger.info(f"¡Modelo {model_name} cargado desde archivo local!")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo desde archivo local: {e}")
            return None
    else:
        logger.error(
            f"Modelo {model_name} no encontrado ni en S3 ni localmente en {local_path}"
        )
        return None


def load_models():
    """Carga los modelos de clasificación y segmentación desde S3 o local"""
    global model_clasificacion, model_segmentacion

    # Cargar modelo de clasificación
    model_clasificacion = load_model_from_s3_or_local(
        MODEL_CLASSIFICATION_NAME,
        f"{S3_MODELS_PREFIX.rstrip('/')}/{MODEL_CLASSIFICATION_NAME}",
    )

    # Cargar modelo de segmentación
    model_segmentacion = load_model_from_s3_or_local(
        MODEL_SEGMENTATION_NAME,
        f"{S3_MODELS_PREFIX.rstrip('/')}/{MODEL_SEGMENTATION_NAME}",
        custom_objects=CUSTOM_OBJECTS,
    )

    return model_clasificacion, model_segmentacion


# Cargar modelos al importar el módulo
load_models()
