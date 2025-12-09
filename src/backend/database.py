import sqlite3
from flask import g

DB_FILE = "hospital_data.db"


def get_db():
    """Obtiene la conexión a la base de datos"""
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_FILE)
        db.row_factory = sqlite3.Row
    return db


def close_connection(exception):
    """Cierra la conexión a la base de datos al finalizar el contexto"""
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db(app):
    """Inicializa la base de datos creando las tablas necesarias"""
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                date TEXT,
                filename TEXT,
                predicted_class TEXT,
                confidence REAL,
                mask_base64 TEXT
            )
        """)
        db.commit()
