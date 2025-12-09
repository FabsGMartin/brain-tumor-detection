# -----------LIBRARIES LOADING-------------

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
import datetime
import requests
import base64
import io
import numpy as np

# ---------- PATH CONSTANTS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
IMAGES_DIR = BASE_DIR / "img"
VIDEO_PATH = BASE_DIR / "video.mp4"
TEAM_IMAGES_DIR = BASE_DIR / "images" / "team"

# CSV files
ROUTES_LABEL_CSV = DATA_DIR / "routes_label.csv"
SEGMENTATION_ROUTES_LABELS_CSV = DATA_DIR / "segmentation_routes_labels.csv"

# Image files
KAGGLE_IMAGE = IMAGES_DIR / "kaggle.png"
TCIA_IMAGE = IMAGES_DIR / "TCIA.png"

# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- CUSTOM CSS STYLING ----------

st.markdown(
    """
    <style>
    /* Highlight analysis box */
    .highlight-box {
        border-left: 6px solid #0747d4;
        background-color: #F0F2F6;
        color: black;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14.5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------DATAFRAME AND MODEL LOADING-------------

df_routes_label = pd.read_csv(str(ROUTES_LABEL_CSV))
df_tumors = pd.read_csv(str(SEGMENTATION_ROUTES_LABELS_CSV))


def call_flask_model(api_url: str, pil_image: Image.Image):
    pil_image = pil_image.convert("RGB")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    url = api_url.rstrip("/") + "/predict"

    resp = requests.post(url, json={"image_base64": img_b64}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def decode_mask_from_b64(mask_b64: str) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    return np.array(mask_img)


def page_home():
    st.markdown("")


def page_intro():
    st.header("🧠 Brain tumor detection and segmentation")

    st.error(
        "- Around 80% of people living with a brain tumor require neurorehabilitation.\n"
        "- In Spain, more than 5,000 new brain tumor cases are diagnosed every year.\n"
        "- Brain tumors account for approximately 2% of all cancers diagnosed in adults and 15% of those diagnosed in children.\n"
        "- About 80% of patients will present cognitive dysfunction, and 78% will present motor dysfunction.\n"
        "- Therapeutic exercise can reduce cancer-related mortality by up to 59%."
    )

    if logo is not None:
        st.image(logo, width=120)

    st.markdown("## Introduction to the clinical problem")
    st.warning(
        "Brain cancer, and in particular low-grade gliomas (LGG), "
        "requires early diagnosis and careful monitoring. "
        "Magnetic resonance imaging (MRI) allows us to visualize the tumor, "
        "but manual delineation is slow and highly dependent on the specialist."
    )

    st.markdown("---")
    st.markdown(
        "In this demo you will see:\n"
        "- Statistics of the patient cohort.\n"
        "- Example positive and negative cases.\n"
        "- How a deep learning model can predict whether an MRI contains a tumor."
    )

    st.info(
        """
        From a clinical perspective, low-grade gliomas often affect relatively young
        adults and may present with seizures, headaches or subtle cognitive changes.
        Even though they are classified as "low grade", they can progress to
        high-grade gliomas, so longitudinal monitoring with MRI and, when indicated,
        histopathological and molecular analysis (e.g. IDH mutation, 1p/19q codeletion)
        are key for prognosis and treatment planning.
        """
    )

    st.markdown(
        """
        For radiologists and data scientists, MRI is interesting because it combines:
        - **Anatomical detail** (T1- and T2-weighted sequences).
        - **Edema and tumor extent** visualization (FLAIR).
        - In some protocols, **functional information** such as diffusion and perfusion,
          which can correlate with cell density and vascularity.
        Integrating these heterogeneous sources of information is one of the main
        motivations for using deep learning in neuro-oncology.
        """
    )


def page_model():
    st.header("🧬 Deep learning model")
    st.markdown(
        """
        In a real hospital workflow, such a model would typically act as a
        **decision-support tool** or “second reader”. It can:
        - Highlight suspicious regions that deserve closer inspection.
        - Provide quantitative measurements (e.g. tumor volume).
        - Help standardize reports across radiologists.
        Final responsibility for diagnosis and treatment decisions always remains
        with the clinical team.
        """
    )

    st.markdown("## General architecture")
    st.markdown(
        """
        Our medical AI system is based on a **deep learning model** that operates on
        brain MRI slices.

        At a high level, the pipeline is:

        1. **Input**: MRI image (normalized and resized).
        2. **Neural network** (e.g. U-Net or CNN):
           - Extracts visual patterns (edges, textures, hyperintense regions...).
           - Learns to distinguish between healthy tissue and tumor tissue.
        3. **Output**:
           - A **class prediction**: tumor / no tumor.
           - Optionally, a **segmentation mask** highlighting tumor pixels.
        """
    )

    st.markdown(
        """
        Although this demo focuses on 2D slices, many research systems work with:
        - **3D convolutions**, which exploit volumetric context across slices.
        - **Multi-sequence input** (T1, T1+contrast, T2, FLAIR) stacked as channels.
        - **Multimodal fusion**, combining imaging with clinical variables
          (age, performance status, molecular markers) or even genomics.
        This richer input can improve performance for tasks such as grading or prognosis.
        """
    )

    st.markdown("## Training (summary)")
    st.markdown(
        """
        - **Data**: MRI dataset with tumor annotations.
        - **Labels**:
          - For classification: `0` = no tumor, `1` = tumor.
          - For segmentation: masks where each pixel indicates tumor/no tumor.
        - **Procedure**:
          - Split into *train / validation / test*.
          - Train for several epochs minimizing a loss function
            (for example, *Binary Cross-Entropy* for classification or
            *Dice loss* for segmentation).
        - **Typical metrics**:
          - Classification: accuracy, F1, sensitivity, specificity.
          - Segmentation: Dice coefficient, IoU.
        """
    )

    st.markdown("## Data preprocessing and quality control")
    st.markdown(
        """
        Before training any medical imaging model, a robust preprocessing pipeline is essential:
        - **Skull stripping** to remove non-brain tissue and reduce noise.
        - **Intensity normalization** per scan to mitigate scanner- or protocol-related variability.
        - **Spatial registration** to a common template when combining data from multiple patients.
        - **Resampling to isotropic voxels** so that physical distances are comparable.
        - **Data augmentation** (rotations, flips, elastic deformations, mild intensity shifts)
          to improve generalization and simulate real-world acquisition variability.
        A careful visual QC (quality control) step is usually performed with radiologists
        to exclude corrupted or mislabeled scans.
        """
    )

    st.markdown("## Evaluation and clinical interpretation")
    st.markdown(
        """
        Beyond global metrics, clinicians and data scientists typically:
        - Inspect **ROC and precision-recall curves** to select thresholds that balance
          sensitivity (avoiding missed tumors) and specificity (avoiding unnecessary alarms).
        - Use **calibration curves** to verify that predicted probabilities correspond
          to actual risk, which is crucial when communicating risk to patients.
        - Analyze **confusion matrices** stratified by subgroups (age, sex, scanner type,
          tumor location) to detect potential bias.
        - Compare performance with human experts in **reader studies** and investigate
          cases where the model disagrees with the radiologist.
        - Perform **external validation** on data from other hospitals to test
          generalization beyond the training cohort.
        """
    )

    st.markdown("## Integration with Flask")
    st.info(
        """
        The model is deployed inside a **Flask API**:

        - The Flask app exposes an HTTP endpoint (for example, `/predict`).
        - Streamlit sends the MRI image to the endpoint in base64 format.
        - Flask runs the deep learning model and returns:
          - whether there is a tumor (`has_tumor`)
          - the probability (`probability`)
          - optionally, a mask (`mask_base64`).

        This separation allows us to:
        - Scale the model independently (GPU/CPU).
        - Use Streamlit only as a lightweight visual interface.
        """
    )

    st.markdown(
        """
        In a production setting, this architecture would be complemented with:
        - **Authentication and audit logs** to track who requested each prediction.
        - **Versioning** of models and training datasets to ensure reproducibility.
        - **Monitoring** of latency, error rates and data drift to detect when
          the model may need to be re-evaluated or retrained.
        - Integration with hospital systems (PACS/RIS) using standards such as DICOM and HL7/FHIR.
        """
    )

    st.markdown("## Limitations and responsible use")
    st.warning(
        """
        This application is a **proof of concept** (PoC):

        - It does not replace the judgment of a medical professional.
        - Predictions may contain errors.
        - Any real clinical use must undergo rigorous validation.
        """
    )

    st.info(
        """
        Even models that perform well in retrospective studies can fail once deployed
        if the patient population, scanners or imaging protocols change over time.
        Continuous surveillance, periodic re-validation and collaboration between
        data scientists, clinicians and MLOps engineers are essential for safe,
        responsible AI in healthcare.
        """
    )


def page_sources():
    st.markdown("""
    ## Dataset Description — LGG MRI Segmentation

    The **LGG MRI Segmentation** dataset comes from the TCGA-LGG collection hosted on [*The Cancer Imaging Archive (TCIA)*](https://www.cancerimagingarchive.net/collection/tcga-lgg/) and was curated and released on [Kaggle by Mateusz Buda](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data). It contains MRI scans of patients diagnosed with **low-grade gliomas**, along with expert-annotated **tumor segmentation masks**.

    ### Key Characteristics
    - **Patients:** ~110
    - **Total images:** ~3,900 MRI slices
    - **Modalities:** Multi-channel `.tif` images (commonly including FLAIR and contrast variations)
    - **Annotations:** Single-channel masks marking the tumor region
    - **Structure:** Each patient folder includes MRI slices and corresponding segmentation masks

    ### Why It’s Useful for Brain Tumor Segmentation
    - Provides **reliable ground-truth labels** for supervised learning.
    - Includes **multiple slices per patient**, giving models diverse anatomical variation.

    """)

    col1, col2, col3, col4, col5 = st.columns(
        [2, 5, 2, 5, 2], gap="large", vertical_alignment="center"
    )
    with col2:
        with st.container(border=True):
            st.image(str(KAGGLE_IMAGE), use_container_width=True)
    with col4:
        with st.container(border=True):
            st.image(str(TCIA_IMAGE), use_container_width=True)


def page_dataset():
    st.header("📊 Database analysis")

    if df.empty:
        st.error(
            "`data.csv` was not found. Place it next to `app.py` and reload the page."
        )
        return

    st.caption(f"Rows: {df.shape[0]} · Columns: {df.shape[1]}")

    st.markdown(
        """
        From an epidemiological and data-science point of view, this dataset represents
        a simplified cohort. In real projects we would also collect variables such as:
        - Age, presenting symptoms and performance status.
        - Tumor grade, histology and molecular markers (e.g. IDH, MGMT, 1p/19q).
        - Treatment (surgery, radiotherapy, chemotherapy, targeted therapies).
        - Outcomes such as progression-free and overall survival.
        These additional features enable models not only for **detection**, but also for
        **prognosis**, **treatment selection** and **response evaluation**.
        """
    )

    tab_tabla, tab_graficas = st.tabs(["📄 Table", "📈 Charts"])

    with tab_tabla:
        st.subheader("Dataset overview")
        st.dataframe(df)

    with tab_graficas:
        if GENDER_COL not in df.columns:
            st.info(f"The column `{GENDER_COL}` was not found in the CSV.")
            return

        st.markdown("### Distribution by gender")

        df_count = df.groupby(GENDER_COL).size().reset_index(name="count")

        fig_pie = px.pie(
            df_count,
            values="count",
            names=GENDER_COL,
            title="Distribution of patients by gender",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.caption(
            "Differences in gender distribution can reflect real epidemiological trends, "
            "but they may also be influenced by sample size, referral patterns or "
            "inclusion criteria of the study."
        )

        if TUMOR_COL in df.columns:
            st.markdown("### Tumor probability by gender")

            df_avg = (
                df.groupby(GENDER_COL)[TUMOR_COL].mean().reset_index(name="Tumor_Prob")
            )

            fig_bar = px.bar(
                df_avg,
                x=GENDER_COL,
                y="Tumor_Prob",
                title="Average tumor probability by gender",
                labels={"Tumor_Prob": "Tumor probability"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.caption(
                "Gender is usually not sufficient to make individual predictions by itself, "
                "but it can be useful to describe the cohort and to check that the model's "
                "performance is not systematically worse in a given subgroup."
            )

            st.markdown("### Query by gender")
            genders = df[GENDER_COL].dropna().unique().tolist()
            sel_gender = st.selectbox("Select gender", genders)

            prob_sel = df_avg.loc[df_avg[GENDER_COL] == sel_gender, "Tumor_Prob"].values
            if len(prob_sel) > 0:
                st.success(
                    f"Estimated average tumor probability for **{sel_gender}**: "
                    f"**{prob_sel[0] * 100:.2f}%**"
                )

            st.markdown("### Global class distribution (tumor vs no tumor)")

            class_counts = df[TUMOR_COL].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]

            fig_bool = px.bar(
                class_counts,
                x="Class",
                y="Count",
                title="Number of patients per class (0 = no tumor, 1 = tumor)",
                text="Count",
            )
            st.plotly_chart(fig_bool, use_container_width=True)

            st.caption(
                "If one class is much more frequent than the other (class imbalance), "
                "we may need strategies such as re-weighting, resampling or specialized "
                "loss functions to avoid a model that simply predicts the majority class."
            )
        else:
            st.info(
                f"The column `{TUMOR_COL}` was not found to compute probabilities "
                "or the class distribution."
            )


def page_cases():
    st.header("🖼️ Ejemplos de tumores cerebrales en RM")

    st.markdown(
        """
        Aquí mostramos cortes de **resonancia magnética cerebral** con y sin **tumor segmentado**.
        En cada ejemplo verás:

        1. **RM original**
        2. **Máscara binaria del tumor** (blanco = tumor, negro = fondo)
        3. **RM con la máscara superpuesta** (solo en los casos con tumor)
        """
    )

    rows_dir = IMAGES_DIR

    # ------------------ CASOS CON TUMOR (row_*.png) ------------------
    tumor_rows = sorted(rows_dir.glob("row_*.png"))

    # ------------------ CASOS SIN TUMOR (example_no_tumor*.png) ------------------
    no_tumor_rows = sorted(rows_dir.glob("example_no_tumor*.png"))

    if not tumor_rows and not no_tumor_rows:
        st.error(
            "No se han encontrado imágenes ni `row_*.png` (con tumor) "
            "ni `example_no_tumor*.png` (sin tumor) en la carpeta "
            f"`{rows_dir}`."
        )
        return

    # ------------------Contenedor central------------------

    left_empty, center, right_empty = st.columns([1, 4, 1])
    with center:
        st.markdown("<br>", unsafe_allow_html=True)

        # Primero SIN tumor, luego CON tumor
        tipo = st.radio(
            "Selecciona tipo de caso",
            ("🟢 Sin tumor", "🔴 Con tumor"),
            horizontal=True,
            index=0,
        )

        if tipo == "🔴 Con tumor":
            active_rows = tumor_rows
            state_key = "random_row_idx_tumor"
            boton_texto = "🔀 Mostrar otro caso con tumor"
            titulo_prefix = "Caso"
            subtitulo_suffix = "tumor cerebral segmentado"
        else:
            active_rows = no_tumor_rows
            state_key = "random_row_idx_no_tumor"
            boton_texto = "🔀 Mostrar otro caso sano"
            titulo_prefix = "Caso sano"
            subtitulo_suffix = "RM sin tumor visible"

        if not active_rows:
            if tipo == "🔴 Con tumor":
                st.warning(
                    "No hay imágenes `row_*.png` para casos con tumor.\n"
                    "Asegúrate de que `row_01.png`, `row_02.png`, ... están en la carpeta Imagen."
                )
            else:
                st.warning(
                    "No hay imágenes `example_no_tumor*.png` para casos sin tumor.\n"
                    "Coloca archivos como `example_no_tumor.png`, "
                    "`example_no_tumor2.png`, ... en la carpeta Imagen."
                )
            return

        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        # Botón centrado
        bc1, bc2, bc3 = st.columns([1, 2, 1])
        with bc2:
            if st.button(boton_texto):
                st.session_state[state_key] = random.randrange(len(active_rows))

        st.markdown("<br>", unsafe_allow_html=True)

        current_idx = st.session_state[state_key]
        current_path = active_rows[current_idx]

        stem = current_path.stem
        num_part = "".join(ch for ch in stem if ch.isdigit())
        case_number = num_part if num_part else "–"

        st.markdown(
            f"<h3 style='text-align:center'>{titulo_prefix} {case_number}: {subtitulo_suffix}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ------------------Mostrar imágenes------------------

        if tipo == "🔴 Con tumor":
            # fila row_XX con 3 columnas en una misma imagen
            img_row = Image.open(current_path)
            w, h = img_row.size
            col_w = w // 3

            img_mri = img_row.crop((0, 0, col_w, h))
            img_mask = img_row.crop((col_w, 0, 2 * col_w, h))
            img_mri_mask = img_row.crop((2 * col_w, 0, w, h))

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    "<h5 style='text-align:center'>RM original</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mri, use_column_width=True)

            with c2:
                st.markdown(
                    "<h5 style='text-align:center'>Máscara de tumor</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mask, use_column_width=True)

            with c3:
                st.markdown(
                    "<h5 style='text-align:center'>RM con máscara</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mri_mask, use_column_width=True)

            st.markdown(
                """
                #### Clinical / data analyst interpretation (with tumor)

                - **Region of interest:** a focal hyperintense lesion is visible within the brain
                  parenchyma. The binary mask highlights all pixels classified as tumor.
                - **Segmentation concept:** every white pixel in the mask corresponds to voxels
                  that the model (or the manual annotation) considers part of the tumor.
                - **Visual benefit:** the overlaid image makes it easier to appreciate tumor
                  borders, mass effect and relationship to surrounding tissue.
                - **From a data point of view:** this slice would be labelled as a **positive
                  sample**, and the mask provides dense supervision for training segmentation
                  models (Dice, IoU, pixel-wise accuracy, etc.).
                """
            )

        else:
            # ejemplo sin tumor: una única RM; la máscara está ya implícita (vacía)
            img_mri = Image.open(current_path).convert("RGB")

            # Pequeño toggle de vista, pero la imagen es la misma
            vista = st.radio(
                "Vista del caso sano",
                ("🧼 Ver RM sin máscara", "🧼 Ver RM con máscara (sin tumor)"),
                horizontal=True,
                key="vista_sano",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if vista == "🧼 Ver RM sin máscara":
                st.markdown(
                    "<h5 style='text-align:center'>RM original (sin tumor)</h5>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<h5 style='text-align:center'>RM con máscara (máscara vacía)</h5>",
                    unsafe_allow_html=True,
                )

            # En ambos casos se muestra la misma imagen, porque no hay tumor
            st.image(img_mri, use_column_width=False)

            # 🧼 Descripción clínico–analítica en inglés (sin tumor)
            st.markdown(
                """
                #### Clinical / data analyst interpretation (no visible tumor)

                - **Overall impression:** normal-appearing brain MRI for this slice, with
                  no focal mass, no clear edema pattern and preserved global symmetry.
                - **Segmentation point of view:** this is a **negative sample**; the
                  corresponding mask is empty, meaning no pixels are labelled as tumor.
                - **Why it matters for the model:** negative cases are crucial to reduce
                  false positives and to teach the network what healthy anatomy looks like.
                - **Expected behavior:** the model should assign low tumor probability to
                  all pixels in this image. Any high activation here would be a potential
                  false positive.
                """
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if tipo == "🔴 Con tumor":
            note_text = (
                "Nota: estos son ejemplos de cortes 2D con tumor. En la práctica se analizan "
                "volúmenes 3D y múltiples secuencias (T1, T2, FLAIR, contraste), junto con "
                "la historia clínica."
            )
        else:
            note_text = (
                "Nota: en estos casos no hay tumor en el corte mostrado. La 'máscara' es vacía, "
                "por lo que la RM con y sin máscara se ven iguales. Compararlos con los casos "
                "con tumor ayuda a entrenar y validar el modelo."
            )

        st.markdown(
            f"<p style='text-align:center; font-size:0.9rem;'>{note_text}</p>",
            unsafe_allow_html=True,
        )


def page_media():
    st.header("🎥 Visual demo and appointment")

    st.subheader("Demo video of the app / model")
    try:
        with open(str(VIDEO_PATH), "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception:
        st.info(
            f"Place a `video.mp4` file at `{VIDEO_PATH}` or update the path in the code."
        )

    st.markdown(
        """
        In an integrated hospital environment, a similar interface could be embedded
        into the radiology workstation or electronic health record to:
        - Visualize AI-generated segmentations directly on the clinician's screen.
        - Provide structured summaries of tumor burden over time.
        - Suggest standardized follow-up intervals according to risk.
        For now, this demo focuses on showing the core concepts in an accessible way.
        """
    )


def page_team():
    st.header("👥 Project team")

    st.markdown(
        """
        This work has been developed by a multidisciplinary team of students
        in Data Science and backend development.

        Below you can see our profiles and GitHub links.
        """
    )

    st.markdown(
        """
        The project combines:
        - **Medical and domain knowledge**, to formulate clinically relevant questions.
        - **Machine learning and MLOps**, to design, train and deploy robust models.
        - **Data engineering**, to process raw DICOM images into analysis-ready tensors.
        - **Frontend and UX design**, to create interfaces that fit real clinical workflows.
        Effective AI in healthcare always requires this kind of cross-disciplinary collaboration.
        """
    )


def page_team():
    st.header("👥 Project team")

    st.markdown(
        """
        This work has been developed by a multidisciplinary team of students
        in Data Science and backend development.

        Below you can see our profiles and GitHub links.
        """
    )

    team = [
        {
            "name": "Luna Pérez",
            "github": "https://github.com/LunaPerezT",
            "photo": str(TEAM_IMAGES_DIR / "miembro1.jpg"),
        },
        {
            "name": "Raquel Hernández",
            "github": "https://github.com/RaquelH18",
            "photo": str(TEAM_IMAGES_DIR / "miembro2.jpg"),
        },
        {
            "name": "Marcos Marín",
            "github": "https://github.com/mmarin3011-cloud",
            "photo": str(TEAM_IMAGES_DIR / "miembro3.jpg"),
        },
        {
            "name": "Fabián G. Martín",
            "github": "https://github.com/FabsGMartin",
            "photo": str(TEAM_IMAGES_DIR / "miembro4.jpg"),
        },
        {
            "name": "Miguel J. de la Torre",
            "github": "https://github.com/migueljdlt",
            "photo": str(TEAM_IMAGES_DIR / "miembro5.jpg"),
        },
        {
            "name": "Alejandro C.",
            "github": "https://github.com/alc98",
            "photo": str(TEAM_IMAGES_DIR / "miembro6.jpg"),
        },
    ]

    # Grid de 2 filas x 3 columnas, con GitHub justo debajo del nombre
    for row_start in range(0, len(team), 3):
        cols = st.columns(3)
        for col, member in zip(cols, team[row_start : row_start + 3]):
            with col:
                st.markdown("### " + member["name"])
                st.markdown(f"**GitHub:** [{member['github']}]({member['github']})")
                try:
                    img = Image.open(member["photo"])
                    st.image(img, use_column_width=True, caption=member["name"])
                except Exception:
                    st.info(f"Photo not found: `{member['photo']}`")

    st.info(
        """
        Although this is an academic project, any real-world deployment as a clinical
        tool would require collaboration with neuroradiologists, neurosurgeons,
        oncologists, medical physicists and hospital IT teams, as well as regulatory
        approval as a medical device.
        """
    )


# ---------- APP HEADER ----------


st.markdown(
    """<h1 style="text-align: center;color: black;"> <b> Brain MRI Tumor Detection </b></h1>
    <h5 style="text-align: center;color: gray"> <em> A Deep Learning based project to detect and segmetate brain tumors in MRI images </em> </h5>""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------- SIDEBAR NAVIGATION ----------

st.sidebar.header("Navigation Menu")
st.sidebar.caption("Choose a section to explore the project.")

menu = [
    "🏠 Home📚 Introduction",
    "📂 Data Sources",
    "🧬 Deep learning model",
    "📊 Data Visualization",
    "🖼️ Example cases",
    "🔍 Live prediction",
    "🎥 Media and appointment",
    "👥 About the Authors",
]

choice = st.sidebar.radio("Select a page:", menu)

# ---------- APP BODY ----------

if choice == "🏠 Home":
    page_home()
elif choice == "📚 Introduction":
    page_intro()
elif choice == "📂 Data Sources":
    page_sources()
elif choice == "🧬 Deep learning model":
    page_model()
elif choice == "📊 Data Visualization":
    page_dataset()
elif choice == "🖼️ Example cases":
    page_cases()
elif choice == "🔍 Live prediction":
    page_live_prediction()
elif choice == "🎥 Media and appointment":
    page_media()
elif choice == "👥 About the Authors":
    page_team()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 1em;'>© 2025 Brain MRI Tumor Detection </p>",
    unsafe_allow_html=True,
)
