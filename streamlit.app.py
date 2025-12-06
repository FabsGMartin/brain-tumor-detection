import streamlit as st 
import pandas as pd
import plotly.express as px
from PIL import Image
import datetime
import requests
import base64
import io
import numpy as np

try:
    logo = Image.open("logo.png")
    page_icon = logo
except Exception:
    logo = None
    page_icon = "🧠"

st.set_page_config(
    page_title="Brain MRI Tumor App",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_data(path: str = "data.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame()
    return df

df = load_data()

GENDER_COL = "Gender"
TUMOR_COL = "Tumor"

def call_flask_model(api_url: str, pil_image: Image.Image):
    pil_image = pil_image.convert("RGB")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    url = api_url.rstrip("/") + "/predict"

    resp = requests.post(
        url,
        json={"image_base64": img_b64},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()

def decode_mask_from_b64(mask_b64: str) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    return np.array(mask_img)

def page_intro():
    st.header("🧠 Detección y segmentación de tumores cerebrales")

    if logo is not None:
        st.image(logo, width=120)

    st.markdown("## Introducción al problema clínico")
    st.warning(
        "El cáncer cerebral, y en particular los gliomas de bajo grado (LGG), "
        "requiere un diagnóstico precoz y una monitorización cuidadosa. "
        "Las resonancias magnéticas (MRI) permiten visualizar el tumor, "
        "pero la delimitación manual es lenta y dependiente del especialista."
    )

    st.markdown("---")
    st.markdown(
        "En esta demo verás:\n"
        "- Estadísticas de la cohorte de pacientes.\n"
        "- Ejemplos de casos positivos y negativos.\n"
        "- Cómo un modelo de deep learning puede predecir si una MRI tiene tumor."
    )

def page_model():
    st.header("🧬 Modelo de deep learning")

    st.markdown("## Arquitectura general")
    st.markdown(
        """
        Nuestro sistema de IA médica se basa en un **modelo de deep learning** que trabaja con
        cortes de resonancia magnética (MRI) del cerebro.

        A alto nivel, el flujo es:

        1. **Entrada**: imagen de MRI (normalizada y redimensionada).
        2. **Red neuronal** (p.ej. U-Net o CNN):
           - Extrae patrones visuales (bordes, texturas, regiones hiperintensas...).
           - Aprende a distinguir entre tejido sano y tejido tumoral.
        3. **Salida**:
           - Una **predicción de clase**: tiene tumor / no tiene tumor.
           - Opcionalmente, una **máscara de segmentación** que marca los píxeles tumorales.
        """
    )

    st.markdown("## Entrenamiento (resumen)")
    st.markdown(
        """
        - **Datos**: dataset de resonancias MRI con anotaciones de tumor.
        - **Etiquetas**:
          - Para clasificación: `0` = sin tumor, `1` = con tumor.
          - Para segmentación: máscaras donde cada píxel indica tumor/no tumor.
        - **Procedimiento**:
          - División en *train / validation / test*.
          - Entrenamiento por épocas minimizando una función de pérdida
            (por ejemplo, *Binary Cross-Entropy* para clasificación o
            *Dice loss* para segmentación).
        - **Métricas típicas**:
          - Clasificación: accuracy, F1, sensibilidad, especificidad.
          - Segmentación: Dice coefficient, IoU.
        """
    )

    st.markdown("## Integración con Flask")
    st.info(
        """
        El modelo está desplegado dentro de una **API Flask**:

        - La app Flask expone un endpoint HTTP (por ejemplo, `/predict`).
        - Streamlit envía la imagen MRI al endpoint en formato base64.
        - Flask ejecuta el modelo de deep learning y devuelve:
          - si hay tumor o no (`has_tumor`)
          - la probabilidad (`probability`)
          - opcionalmente, una máscara (`mask_base64`).

        Esta separación permite:
        - Escalar el modelo de forma independiente (GPU/CPU).
        - Usar Streamlit solo como interfaz visual ligera.
        """
    )

    st.markdown("## Limitaciones y uso responsable")
    st.warning(
        """
        Esta aplicación es una **prueba de concepto** (PoC):

        - No sustituye el criterio de un profesional médico.
        - Las predicciones pueden contener errores.
        - Cualquier uso clínico real debe pasar por validaciones rigurosas.
        """
    )

def page_dataset():
    st.header("📊 Análisis de la base de datos")

    if df.empty:
        st.error("No se ha encontrado `data.csv`. Colócalo junto a `app.py` y recarga la página.")
        return

    st.caption(f"Filas: {df.shape[0]} · Columnas: {df.shape[1]}")

    tab_tabla, tab_graficas = st.tabs(["📄 Tabla", "📈 Gráficas"])

    with tab_tabla:
        st.subheader("Vista general del dataset")
        st.dataframe(df)

    with tab_graficas:
        if GENDER_COL not in df.columns:
            st.info(f"No se encontró la columna `{GENDER_COL}` en el CSV.")
            return

        st.markdown("### Distribución por género")

        df_count = df.groupby(GENDER_COL).size().reset_index(name="count")

        fig_pie = px.pie(
            df_count,
            values="count",
            names=GENDER_COL,
            title="Distribución de pacientes por género"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        if TUMOR_COL in df.columns:
            st.markdown("### Probabilidad de tumor por género")

            df_avg = df.groupby(GENDER_COL)[TUMOR_COL].mean().reset_index(name="Tumor_Prob")

            fig_bar = px.bar(
                df_avg,
                x=GENDER_COL,
                y="Tumor_Prob",
                title="Probabilidad media de tumor por género",
                labels={"Tumor_Prob": "Probabilidad de tumor"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### Consulta por género")
            genders = df[GENDER_COL].dropna().unique().tolist()
            sel_gender = st.selectbox("Selecciona género", genders)

            prob_sel = df_avg.loc[df_avg[GENDER_COL] == sel_gender, "Tumor_Prob"].values
            if len(prob_sel) > 0:
                st.success(
                    f"Probabilidad media estimada de tumor para **{sel_gender}**: "
                    f"**{prob_sel[0]*100:.2f}%**"
                )

            st.markdown("### Distribución global de clases (tumor vs no tumor)")

            class_counts = df[TUMOR_COL].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]

            fig_bool = px.bar(
                class_counts,
                x="Class",
                y="Count",
                title="Número de pacientes por clase (0 = no tumor, 1 = tumor)",
                text="Count"
            )
            st.plotly_chart(fig_bool, use_container_width=True)
        else:
            st.info(
                f"No se encontró la columna `{TUMOR_COL}` para calcular probabilidades "
                "ni la distribución de clases."
            )

def page_cases():
    st.header("🖼️ Casos ejemplo: negativo vs positivo")

    st.markdown(
        "En esta sección mostramos un ejemplo de paciente **sin tumor** (caso negativo) "
        "y un paciente **con tumor** (caso positivo), junto con sus máscaras de segmentación."
    )

    neg_img_path = "images/caso_negativo_mri.png"
    neg_mask_path = "images/caso_negativo_mask.png"
    pos_img_path = "images/caso_positivo_mri.png"
    pos_mask_path = "images/caso_positivo_mask.png"

    st.markdown("### Caso negativo (sin tumor)")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("MRI – caso negativo")
        try:
            neg_img = Image.open(neg_img_path)
            st.image(neg_img, use_column_width=True)
        except Exception:
            st.info(f"Coloca la imagen del caso negativo en `{neg_img_path}`.")

    with col2:
        st.caption("Máscara – caso negativo (sin tumor)")
        try:
            neg_mask = Image.open(neg_mask_path)
            st.image(neg_mask, use_column_width=True)
        except Exception:
            st.info(f"Coloca la máscara del caso negativo en `{neg_mask_path}`.")

    st.markdown("---")
    st.markdown("### Caso positivo (con tumor)")
    col3, col4 = st.columns(2)

    with col3:
        st.caption("MRI – caso positivo")
        try:
            pos_img = Image.open(pos_img_path)
            st.image(pos_img, use_column_width=True)
        except Exception:
            st.info(f"Coloca la imagen del caso positivo en `{pos_img_path}`.")

    with col4:
        st.caption("Máscara – caso positivo (tumor en rojo)")
        try:
            pos_mask = Image.open(pos_mask_path)
            st.image(pos_mask, use_column_width=True)
        except Exception:
            st.info(f"Coloca la máscara del caso positivo en `{pos_mask_path}`.")

def page_live_prediction():
    st.header("🔍 Predicción en vivo con modelo Flask")

    st.markdown(
        """
        Sube una imagen de MRI y el sistema consultará al **modelo de deep learning**
        desplegado en Flask para predecir si hay tumor o no.
        """
    )

    st.sidebar.markdown("### ⚙️ Configuración de la API Flask")
    api_url = st.sidebar.text_input("URL base de la API", "http://localhost:8000")

    uploaded_file = st.file_uploader(
        "Sube una imagen MRI (PNG/JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="MRI subida", use_column_width=True)

        if st.button("Analizar MRI"):
            with st.spinner("Consultando modelo en Flask..."):
                try:
                    response = call_flask_model(api_url, pil_img)
                except Exception as e:
                    st.error(f"Error al llamar a la API: {e}")
                    return

            st.markdown("### Resultado del modelo")

            has_tumor = response.get("has_tumor", None)
            prob = response.get("probability", None)

            if has_tumor is None or prob is None:
                st.error(
                    "La respuesta de la API no contiene las claves esperadas "
                    "(`has_tumor`, `probability`). Ajusta el código a tu formato."
                )
            else:
                diagnosis = "TUMOR DETECTADO" if has_tumor else "SIN INDICIOS DE TUMOR"
                color = "🔴" if has_tumor else "🟢"

                st.metric(
                    label="Diagnóstico del modelo",
                    value=f"{color} {diagnosis}"
                )
                st.metric(
                    label="Probabilidad de tumor",
                    value=f"{prob*100:.2f} %"
                )

            mask_b64 = response.get("mask_base64", None)
            if mask_b64:
                st.markdown("### Máscara de segmentación (opcional)")
                try:
                    mask_arr = decode_mask_from_b64(mask_b64)
                    st.image(mask_arr, caption="Máscara predicha por el modelo", use_column_width=True)
                except Exception:
                    st.info("No se pudo decodificar la máscara devuelta por la API.")

def page_media():
    st.header("🎥 Demo visual y cita")

    st.subheader("Imágenes de ejemplo")

    try:
        img_local = Image.open("imagen.png")
        st.image(img_local, caption="Imagen local de ejemplo", use_column_width=True)
    except Exception:
        st.info("Coloca una imagen llamada `imagen.png` junto a `app.py` o cambia la ruta.")

    st.image(
        "https://picsum.photos/1280",
        caption="Imagen de ejemplo desde URL",
        use_column_width=True
    )

    st.subheader("Vídeo demostrativo de la app / modelo")
    try:
        with open("video.mp4", "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception:
        st.info("Coloca un `video.mp4` junto a `app.py` o actualiza la ruta en el código.")

    st.subheader("📅 Simulación de cita")
    cita = st.date_input("Selecciona una fecha para la cita de revisión", datetime.date.today())
    st.success(f"Fecha seleccionada: {cita.strftime('%d/%m/%Y')}")

def page_team():
    st.header("👥 Equipo del proyecto")

    st.markdown(
        """
        Este trabajo ha sido desarrollado por un equipo multidisciplinar de estudiantes
        de Data Science y desarrollo backend.  
        
        A continuación puedes ver nuestros perfiles y enlaces a GitHub.
        """
    )

    team = [
        {
            "name": "Nombre 1",
            "role": "ML Engineer",
            "github": "https://github.com/usuario1",
            "photo": "images/team/miembro1.jpg"
        },
        {
            "name": "Nombre 2",
            "role": "Backend Developer",
            "github": "https://github.com/usuario2",
            "photo": "images/team/miembro2.jpg"
        },
        {
            "name": "Nombre 3",
            "role": "Data Scientist",
            "github": "https://github.com/usuario3",
            "photo": "images/team/miembro3.jpg"
        },
        {
            "name": "Nombre 4",
            "role": "MLOps & Cloud",
            "github": "https://github.com/usuario4",
            "photo": "images/team/miembro4.jpg"
        },
        {
            "name": "Nombre 5",
            "role": "Frontend & UX",
            "github": "https://github.com/usuario5",
            "photo": "images/team/miembro5.jpg"
        },
        {
            "name": "Nombre 6",
            "role": "Data Engineer",
            "github": "https://github.com/usuario6",
            "photo": "images/team/miembro6.jpg"
        },
    ]

    for row_start in range(0, len(team), 3):
        cols = st.columns(3)
        for col, member in zip(cols, team[row_start:row_start + 3]):
            with col:
                st.markdown("### " + member["name"])
                try:
                    img = Image.open(member["photo"])
                    st.image(img, use_column_width=True, caption=member["role"])
                except Exception:
                    st.info(f"Foto no encontrada: `{member['photo']}`")

                st.markdown(
                    f"[🌐 GitHub]({member['github']})",
                    unsafe_allow_html=True
                )

def main():
    st.title("Brain MRI Tumor – Demo Streamlit")

    st.sidebar.header("Navegación")
    st.sidebar.caption("Elige una sección para explorar el proyecto.")

def main():
    st.title("Brain MRI Tumor – Demo Streamlit")

    st.sidebar.header("Navegación")
    st.sidebar.caption("Elige una sección para explorar el proyecto.")

    menu = [
        "🏠 Introducción",
        "🧬 Modelo de deep learning",
        "📊 Base de datos y gráficas",
        "🖼️ Casos ejemplo",
        "🔍 Predicción en vivo",
        "🎥 Multimedia y cita",
        "👥 Equipo"
    ]

    choice = st.sidebar.radio("Selecciona una página:", menu)

    if choice == "🏠 Introducción":
        page_intro()
    elif choice == "🧬 Modelo de deep learning":
        page_model()
    elif choice == "📊 Base de datos y gráficas":
        page_dataset()
    elif choice == "🖼️ Casos ejemplo":
        page_cases()
    elif choice == "🔍 Predicción en vivo":
        page_live_prediction()
    elif choice == "🎥 Multimedia y cita":
        page_media()
    elif choice == "👥 Equipo":
        page_team()

if __name__ == "__main__":
    main()
