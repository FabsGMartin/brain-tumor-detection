"""
Módulo para almacenar y recuperar predicciones en almacenamiento local.
Reemplaza la funcionalidad de S3 con almacenamiento en archivos locales.
"""

import os
import json
import uuid
import logging
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class LocalPredictionStorage:
    """Maneja el almacenamiento de predicciones en archivos locales."""

    def __init__(self, base_dir: Optional[Path] = None, prefix: str = "predictions/"):
        """
        Inicializa el almacenamiento de predicciones local.

        Args:
            base_dir: Directorio base donde se guardarán las predicciones (default: src/backend/)
            prefix: Prefijo para las predicciones (default: "predictions/")
        """
        if base_dir is None:
            # Por defecto, usar el directorio del módulo
            BASE_DIR = Path(__file__).resolve().parent
            self.base_dir = BASE_DIR
        else:
            self.base_dir = Path(base_dir)

        self.prefix = prefix.rstrip("/") + "/"
        self.predictions_dir = self.base_dir / self.prefix
        self.masks_dir = self.predictions_dir / "masks"
        self.index_path = self.predictions_dir / "index.json"

        # Crear directorios si no existen
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Almacenamiento local configurado en: {self.predictions_dir}")

    def _get_prediction_path(self, prediction_type: str, prediction_id: str) -> Path:
        """Genera la ruta local para una predicción."""
        type_dir = self.predictions_dir / prediction_type
        type_dir.mkdir(parents=True, exist_ok=True)
        return type_dir / f"{prediction_id}.json"

    def _get_mask_path(self, prediction_id: str) -> Path:
        """Genera la ruta local para una máscara."""
        return self.masks_dir / f"{prediction_id}.png"

    def _load_index(self) -> Dict[str, Any]:
        """Carga el índice de predicciones desde archivo local."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                logger.debug(
                    f"Índice cargado: {len(index_data.get('predictions', []))} predicciones"
                )
                return index_data
            except Exception as e:
                logger.error(f"Error cargando índice: {e}")
                raise
        else:
            # Índice no existe aún, crear uno vacío
            logger.info("Índice no existe, creando uno nuevo")
            index_data = {
                "predictions": [],
                "last_updated": datetime.now().isoformat(),
            }
            return index_data

    def _save_index(self, index_data: Dict[str, Any]) -> None:
        """Guarda el índice de predicciones en archivo local."""
        try:
            index_data["last_updated"] = datetime.now().isoformat()
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            logger.debug(
                f"Índice guardado con {len(index_data.get('predictions', []))} predicciones"
            )
        except Exception as e:
            logger.error(f"Error guardando índice: {e}")
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
        Guarda una predicción en almacenamiento local.

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
        mask_path = None
        if mask_base64:
            mask_size_kb = len(mask_base64) / 1024
            if mask_size_kb > 75:  # Si es mayor a ~75KB, guardar como archivo separado
                try:
                    mask_bytes = base64.b64decode(mask_base64)
                    mask_path = self._get_mask_path(prediction_id)

                    with open(mask_path, "wb") as f:
                        f.write(mask_bytes)

                    prediction_data["mask_path"] = str(mask_path.relative_to(self.base_dir))
                    logger.info(
                        f"Máscara grande guardada: {mask_path} ({mask_size_kb:.1f}KB)"
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

        # Guardar predicción en archivo local
        prediction_path = self._get_prediction_path(prediction_type, prediction_id)

        try:
            with open(prediction_path, "w", encoding="utf-8") as f:
                json.dump(prediction_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Predicción guardada: {prediction_path}")

            # Añadir al índice
            index_metadata = {
                "id": prediction_id,
                "type": prediction_type,
                "date": timestamp,
                "filename": filename,
                "path": str(prediction_path.relative_to(self.base_dir)),
            }
            if predicted_class:
                index_metadata["predicted_class"] = predicted_class
            if confidence is not None:
                index_metadata["confidence"] = confidence
            if mask_path:
                index_metadata["mask_path"] = str(mask_path.relative_to(self.base_dir))

            self._add_to_index(index_metadata)

            return prediction_id
        except Exception as e:
            logger.error(f"Error guardando predicción: {e}")
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
        prediction_path = self._get_prediction_path(prediction_type, prediction_id)

        if not prediction_path.exists():
            logger.debug(f"Predicción no encontrada: {prediction_path}")
            return None

        try:
            with open(prediction_path, "r", encoding="utf-8") as f:
                prediction_data = json.load(f)

            # Si la máscara está en un archivo separado, cargarla
            if "mask_path" in prediction_data:
                try:
                    mask_full_path = self.base_dir / prediction_data["mask_path"]
                    if mask_full_path.exists():
                        with open(mask_full_path, "rb") as f:
                            mask_bytes = f.read()
                        prediction_data["mask_base64"] = base64.b64encode(
                            mask_bytes
                        ).decode("utf-8")
                    else:
                        logger.warning(
                            f"Archivo de máscara no encontrado: {mask_full_path}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error cargando máscara desde {prediction_data['mask_path']}: {e}"
                    )

            return prediction_data
        except Exception as e:
            logger.error(f"Error obteniendo predicción: {e}")
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
        """Método de compatibilidad - no hay caché en almacenamiento local."""
        pass
