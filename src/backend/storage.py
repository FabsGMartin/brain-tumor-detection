"""
Módulo para almacenar y recuperar predicciones desde S3.
Reemplaza la funcionalidad de SQLite con almacenamiento en S3.
"""

import os
import json
import uuid
import logging
import base64
from datetime import datetime
from typing import Optional, Dict, List, Any
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class S3PredictionStorage:
    """Maneja el almacenamiento de predicciones en S3."""

    def __init__(self, s3_client, bucket_name: str, prefix: str = "predictions/"):
        """
        Inicializa el almacenamiento de predicciones en S3.

        Args:
            s3_client: Cliente boto3 de S3
            bucket_name: Nombre del bucket S3
            prefix: Prefijo para las predicciones (default: "predictions/")
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.index_key = f"{self.prefix}index.json"
        self.index_cache = None
        self.index_loaded = False

    def _get_prediction_key(self, prediction_type: str, prediction_id: str) -> str:
        """Genera la clave S3 para una predicción."""
        return f"{self.prefix}{prediction_type}/{prediction_id}.json"

    def _get_mask_key(self, prediction_id: str) -> str:
        """Genera la clave S3 para una máscara."""
        return f"{self.prefix}masks/{prediction_id}.png"

    def _load_index(self) -> Dict[str, Any]:
        """Carga el índice de predicciones desde S3."""
        if self.index_loaded and self.index_cache is not None:
            return self.index_cache

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=self.index_key
            )
            index_data = json.loads(response["Body"].read().decode("utf-8"))
            self.index_cache = index_data
            self.index_loaded = True
            logger.debug(
                f"Índice cargado desde S3: {len(index_data.get('predictions', []))} predicciones"
            )
            return index_data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                # Índice no existe aún, crear uno vacío
                logger.info("Índice no existe, creando uno nuevo")
                self.index_cache = {
                    "predictions": [],
                    "last_updated": datetime.now().isoformat(),
                }
                self.index_loaded = True
                return self.index_cache
            else:
                logger.error(f"Error cargando índice desde S3: {e}")
                raise
        except Exception as e:
            logger.error(f"Error inesperado cargando índice: {e}")
            raise

    def _save_index(self, index_data: Dict[str, Any]) -> None:
        """Guarda el índice de predicciones en S3."""
        try:
            index_data["last_updated"] = datetime.now().isoformat()
            index_json = json.dumps(index_data, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.index_key,
                Body=index_json.encode("utf-8"),
                ContentType="application/json",
            )
            self.index_cache = index_data
            logger.debug(
                f"Índice guardado en S3 con {len(index_data.get('predictions', []))} predicciones"
            )
        except Exception as e:
            logger.error(f"Error guardando índice en S3: {e}")
            raise

    def _add_to_index(self, prediction_metadata: Dict[str, Any]) -> None:
        """Añade una predicción al índice."""
        index_data = self._load_index()
        predictions = index_data.get("predictions", [])

        # Añadir al inicio de la lista
        predictions.insert(0, prediction_metadata)

        # Mantener solo los últimos 1000 registros en el índice para evitar que crezca demasiado
        if len(predictions) > 1000:
            predictions = predictions[:1000]

        index_data["predictions"] = predictions
        self._save_index(index_data)

    def save_prediction(
        self,
        prediction_type: str,
        filename: str,
        predicted_class: Optional[str] = None,
        confidence: Optional[float] = None,
        mask_base64: Optional[str] = None,
    ) -> str:
        """
        Guarda una predicción en S3.

        Args:
            prediction_type: "clasificacion" o "segmentacion"
            filename: Nombre del archivo original
            predicted_class: Clase predicha (solo para clasificación)
            confidence: Nivel de confianza (solo para clasificación)
            mask_base64: Máscara en base64 (solo para segmentación)

        Returns:
            ID único de la predicción (UUID)
        """
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Preparar datos de la predicción
        prediction_data = {
            "id": prediction_id,
            "type": prediction_type,
            "date": timestamp,
            "filename": filename,
        }

        # Guardar máscara por separado si es grande (>100KB en base64 ≈ 75KB reales)
        mask_key = None
        if mask_base64:
            mask_size_kb = len(mask_base64) / 1024
            if mask_size_kb > 75:  # Si es mayor a ~75KB, guardar como archivo separado
                try:
                    mask_bytes = base64.b64decode(mask_base64)
                    mask_key = self._get_mask_key(prediction_id)

                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=mask_key,
                        Body=mask_bytes,
                        ContentType="image/png",
                    )
                    prediction_data["mask_key"] = mask_key
                    logger.info(
                        f"Máscara grande guardada en S3: {mask_key} ({mask_size_kb:.1f}KB)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error guardando máscara separada, usando inline: {e}"
                    )
                    prediction_data["mask_base64"] = mask_base64
            else:
                prediction_data["mask_base64"] = mask_base64

        if predicted_class:
            prediction_data["predicted_class"] = predicted_class
        if confidence is not None:
            prediction_data["confidence"] = confidence

        # Guardar predicción en S3
        prediction_key = self._get_prediction_key(prediction_type, prediction_id)
        prediction_json = json.dumps(prediction_data, indent=2)

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=prediction_key,
                Body=prediction_json.encode("utf-8"),
                ContentType="application/json",
            )
            logger.info(f"Predicción guardada en S3: {prediction_key}")

            # Añadir al índice
            index_metadata = {
                "id": prediction_id,
                "type": prediction_type,
                "date": timestamp,
                "filename": filename,
                "key": prediction_key,
            }
            if predicted_class:
                index_metadata["predicted_class"] = predicted_class
            if confidence is not None:
                index_metadata["confidence"] = confidence
            if mask_key:
                index_metadata["mask_key"] = mask_key

            self._add_to_index(index_metadata)

            return prediction_id
        except Exception as e:
            logger.error(f"Error guardando predicción en S3: {e}")
            raise

    def get_prediction(
        self, prediction_id: str, prediction_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene una predicción por ID.

        Args:
            prediction_id: ID de la predicción
            prediction_type: "clasificacion" o "segmentacion"

        Returns:
            Diccionario con los datos de la predicción o None si no existe
        """
        prediction_key = self._get_prediction_key(prediction_type, prediction_id)

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=prediction_key
            )
            prediction_data = json.loads(response["Body"].read().decode("utf-8"))

            # Si la máscara está en un archivo separado, cargarla
            if "mask_key" in prediction_data:
                try:
                    mask_response = self.s3_client.get_object(
                        Bucket=self.bucket_name, Key=prediction_data["mask_key"]
                    )
                    mask_bytes = mask_response["Body"].read()
                    prediction_data["mask_base64"] = base64.b64encode(
                        mask_bytes
                    ).decode("utf-8")
                except Exception as e:
                    logger.warning(
                        f"Error cargando máscara desde {prediction_data['mask_key']}: {e}"
                    )

            return prediction_data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug(f"Predicción no encontrada: {prediction_key}")
                return None
            else:
                logger.error(f"Error obteniendo predicción desde S3: {e}")
                raise
        except Exception as e:
            logger.error(f"Error inesperado obteniendo predicción: {e}")
            return None

    def get_history(
        self, limit: int = 10, prediction_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de predicciones.

        Args:
            limit: Número máximo de predicciones a retornar
            prediction_type: Filtrar por tipo ("clasificacion" o "segmentacion")

        Returns:
            Lista de predicciones (solo metadatos, no datos completos)
        """
        index_data = self._load_index()
        predictions = index_data.get("predictions", [])

        # Filtrar por tipo si se especifica
        if prediction_type:
            predictions = [p for p in predictions if p.get("type") == prediction_type]

        # Limitar resultados
        predictions = predictions[:limit]

        # Cargar datos completos solo para las predicciones necesarias
        result = []
        for pred_meta in predictions:
            try:
                full_pred = self.get_prediction(pred_meta["id"], pred_meta["type"])
                if full_pred:
                    result.append(full_pred)
            except Exception as e:
                logger.warning(f"Error cargando predicción {pred_meta['id']}: {e}")
                # Si falla cargar, usar solo metadatos del índice
                result.append(pred_meta)

        return result

    def invalidate_index_cache(self) -> None:
        """Invalida la caché del índice, forzando una recarga."""
        self.index_loaded = False
        self.index_cache = None
