# -----------LIBRARIES LOADING-------------

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path

# import datetime
import requests
import base64
import io
import numpy as np
import random

# ---------- PATH CONSTANTS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
IMAGES_DIR = BASE_DIR / "img"
VIDEO_PATH = BASE_DIR / "video.mp4"

# CSV files
ROUTE_LABEL_CSV = DATA_DIR / "route_label.csv"
SEGMENTATION_ROUTES_LABELS_CSV = DATA_DIR / "segmentation_routes_labels.csv"

# Image files
KAGGLE_IMAGE = IMAGES_DIR / "kaggle.png"
TCIA_IMAGE = IMAGES_DIR / "TCIA.png"
GITHUB_IMAGE = IMAGES_DIR / "github.png"

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

    <style>
    /* Red highlight analysis box */
    .red-box {
        border-left: 6px solid #FF0000;
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

# -----------DATAFRAME LOADING and ROUTES -------------

df = pd.read_csv(str(ROUTE_LABEL_CSV))
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
    st.header("🏠 **Home**")
    st.markdown("")
    st.markdown(
        """
        Welcome to the Brain MRI Tumor Detection webpage!


        This project focuses on the development of a deep learning system for **brain tumor segmentation and detection in MRI scans**, aiming to support medical research and improve early identification of low-grade gliomas.

        <div class="highlight-box">

        The project combines:
        - **Medical and domain knowledge**, to formulate clinically relevant questions.
        - **AI engineering and AIOps**, to design, train and deploy robust models.
        - **Data engineering**, to process raw TIFF images into analysis-ready tensors.
        - **Frontend and UX design**, to create interfaces that fit real clinical workflows.
        Effective AI in healthcare always requires this kind of cross-disciplinary collaboration.

        </div>

        The website is organized into several sections to guide you through the project:
        - 🏠 **Home** – Overview of the project
        - 📚 **Introduction** – Context and motivation
        - 📂 **Data Sources** – Description of the datasets used
        - 🧬 **Deep Learning Model** – Architecture, training, and methodology
        - 📊 **Data Visualization** – Exploratory and technical visual analyses
        - 🔍 **Live Prediction** – Real-time model inference on user-uploaded MRI images
        - 🎥 **Visual Demo** – Practical demonstration of the segmentation results
        - 🤝 **Collaboration** – Ways to contribute to the project or cancer research
        - 👥 **About the Authors** – Information about the project contributors

        Thank you for visiting — your interest and participation help strengthen ongoing efforts in medical imaging and cancer research.

        """,
        unsafe_allow_html=True,
    )


def page_intro():
    st.header("📚 **Introduction**")
    st.markdown(
        """

        <h5 style="text-align: center;color: black;"> <b>What Is a Low-Grade Glioma?</b></h5>

        **Brain cancer**, and in particular **low-grade gliomas (LGG) requires early diagnosis and careful monitoring**.
        From a clinical perspective, low-grade gliomas often affect relatively young adults and may present with **seizures, headaches or subtle cognitive changes**.
        <br> </br>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h5 style="text-align: center;color: black;"> <b>Why Early Detection Is Important?</b></h5>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """

                **Early detection of brain tumors** plays a crucial role in **improving patient outcomes**. When identified at an early stage, tumors are often smaller, less aggressive, and more responsive to treatment, **allowing clinicians to intervene before neurological damage becomes extensive**.

                """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="red-box">

        - Around 80% of people living with a brain tumor require neurorehabilitation.
        - In 2022, 322,000 new cases of brain and central nervous system tumors were estimated globally.
        - Brain tumors account for approximately 2% of all cancers diagnosed in adults and 15% of those diagnosed in children.
        - About 80% of patients will present cognitive dysfunction, and 78% will present motor dysfunction.

        </div>
        <br></br>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h5 style="text-align: center;color: black;"> <b>Why MRI tumor segmentation is important in Low-Grade Glioma Patients?</b></h5>

        Even though they are classified as "low grade", **they can progress to high-grade gliomas**, so **longitudinal monitoring with MRI** and, when indicated, histopathological and molecular analysis are **key for prognosis and treatment planning**.

        **MRI-based diagnosis** is especially valuable, as it **provides detailed structural information without exposing patients to radiation**.

         <div class="highlight-box">

        For radiologists and data scientists, MRI is interesting because it combines:
        - **Anatomical detail** (T1- and T2-weighted sequences).
        - **Edema and tumor extent** visualization (FLAIR).
        - In some protocols, **functional information** such as diffusion and perfusion,
          which can correlate with cell density and vascularity.
        Integrating these heterogeneous sources of information is one of the main
        motivations for using deep learning in neuro-oncology.

         </div>
        <br></br>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """Advances in **automated image analysis and deep learning** now offer the possibility of **supporting radiologists with faster, more consistent tumor identification**. By accelerating the diagnostic process, **reducing human error, and enabling timely intervention**, early detection becomes a powerful tool in improving survival rates and enhancing quality of life for patients affected by brain tumors.")
    """,
        unsafe_allow_html=True,
    )


def page_sources():
    st.header("📂 **Data Sources**")
    st.markdown(
        """

    The **LGG MRI Segmentation** dataset comes from the TCGA-LGG collection hosted on [*The Cancer Imaging Archive (TCIA)*](https://www.cancerimagingarchive.net/collection/tcga-lgg/) and was curated and released on [Kaggle by Mateusz Buda](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data). It contains MRI scans of patients diagnosed with **low-grade gliomas**, along with expert-annotated **tumor segmentation masks**.
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4, col5 = st.columns(
        [2, 5, 2, 5, 2], gap="large", vertical_alignment="center"
    )
    with col2:
        with st.container(
            border=True,
        ):
            st.image(str(KAGGLE_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        Kaggle – [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

                        </center>
                        """,
                unsafe_allow_html=True,
            )
    with col4:
        with st.container(border=True):
            st.image(str(TCIA_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        TCIA – [TCGA-LGG Collection](https://www.cancerimagingarchive.net)

                        </center>
                        """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
    ##### **Data Key Characteristics**

    - **Patients:** ~110
    - **Total images:** ~3,900 MRI slices
    - **Modalities:** Multi-channel `.tiff` images (commonly including FLAIR and contrast variations)
    - **Annotations:** Single-channel masks marking the tumor region
    - **Structure:** Each patient folder includes MRI slices and corresponding segmentation masks

    ##### **Why It’s Useful for Brain Tumor Segmentation?**

    - Provides **reliable ground-truth labels** for supervised learning.
    - Includes **multiple slices per patient**, giving models diverse anatomical variation.

    """,
        unsafe_allow_html=True,
    )


def page_data():
    st.header("📊 Dataset Visualization")
    df_routes = pd.read_csv(str(ROUTE_LABEL_CSV), index_col=0)
    tab_plots, tab_table = st.tabs(["📈 Plots", "📄 Table"])

    # ===== PLOTS =====
    with tab_plots:
        st.subheader("Class distribution")

        # Count 0 and 1
        class_counts = df_routes["mask"].value_counts().reset_index()
        class_counts.columns = ["mask_value", "Number of images"]

        class_counts["Class"] = class_counts["mask_value"].map(
            {
                0: "0 – Negative (no tumor)",
                1: "1 – Positive (tumor present)",
            }
        )

        # Keep only the columns needed for the plot
        class_counts = class_counts[["Class", "Number of images"]]

        # Pie chart
        fig_pie = px.pie(
            class_counts,
            names="Class",
            values="Number of images",
            title="Tumor vs No Tumor",
        )
        col1, col2, col3 = st.columns(3)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)

        # Global prevalence (image-level)
        prevalence = df_routes["mask"].mean()
        st.markdown(
            f"""
                    <div style="text-align: center;">

                    *In this dataset ≈ {prevalence * 100:.2f}% of the images are labelled as positive (`mask = 1`).*

                     </div>

            """,
            unsafe_allow_html=True,
        )

        # ===== TABLE =====
        with tab_table:
            st.subheader("`route_label.csv`")
            st.markdown(
                "*Overview of routes MRI images and mask, with their corresponding label (0 – Negative (no tumor),1 – Positive (tumor present)*"
            )
            st.dataframe(df_routes[df_routes.columns])
    # =====================================================================
    #  🔬 Scientific medical + data science interpretation
    # =====================================================================
    prevalence_global = df_routes["mask"].mean()
    negative_pct = (1 - prevalence_global) * 100
    positive_pct = prevalence_global * 100

    show_analysis = st.expander("Show Authors' Analysis", expanded=True)
    with show_analysis:
        st.markdown(
            f"""

        <h4 style="margin:0 0 8px 0;">✍️ Author’s Analysis</h4>
    <div class="highlight-box">
<p style="margin:0;">


<h5 style="text-align: center;color: black;"> <b> Cohort composition (image-level class distribution)</b></h5>

In this dataset:


- **≈ {negative_pct:.1f}%** of MRI slices are labelled as
  **0 – Negative (no tumor)**
- **≈ {positive_pct:.1f}%** of MRI slices are labelled as
  **1 – Positive (tumor present)**

This yields an **image-level tumor prevalence of approximately {positive_pct:.1f}%**.

From a methodological standpoint, this indicates a **moderately imbalanced dataset**,
with a dominant negative class and a substantial proportion of positive slices.
Therefore, **any classification model** must outperform a trivial baseline predicting
the majority class (≈ **{negative_pct:.1f}% accuracy**) to demonstrate meaningful discriminative value.


<h5 style="text-align: center;color: black;"> <b> Clinical and machine-learning implications </b></h5>

- The enrichment in tumor-positive slices (≈ {positive_pct:.1f}%) is higher than in routine clinical cohorts,
  which usually contain far fewer tumors. This is advantageous for model development, as it provides a
  sufficient number of positive examples to learn tumor-related patterns and to train segmentation models.

- Because of the moderate class imbalance, evaluation should not rely solely on accuracy. More informative metrics are:

  - **Sensitivity / recall** for positive cases (`mask = 1`)
  - **Specificity** for negative cases (`mask = 0`)
  - **AUC-ROC** and **AUC-PR**, which better capture performance under imbalance.

- If the model tends to under-detect tumors, one may consider:
  - **Class-weighted loss functions**
  - **Focal loss**
  - **Oversampling of positive slices** or undersampling of negatives.


<h5 style="text-align: center;color: black;"> <b>

Utility of the `mask` column
</b></h5>
Although voxel-wise segmentation masks are available via `mask_path`, the binary image-level label (`mask`) enables:


- Rapid assessment of **class distribution** (as visualised in the pie chart).
- Training of a **binary tumor vs. no-tumor classifier** as a screening or pre-filtering stage.
- Stratified analyses, for example comparing intensity distributions or radiomic features between positive and negative slices.

</p>
<div/>

From a clinical research perspective, the cohort can be succinctly described as:

> *"In this dataset, approximately {positive_pct:.1f}% of MRI slices contain visible tumor tissue according to expert segmentation. This prevalence establishes the baseline that any automated detection model must exceed in order to be clinically relevant."*


""",
            unsafe_allow_html=True,
        )


def page_cases():
    st.header("🧠 MRI Images Visualization")

    st.markdown(
        """
        Here we show slices of **brain magnetic resonance imaging (MRI)** with and without **segmented tumor**.
        In each example you will see:

        1. **Original MRI**
        2. **Binary tumor mask** (white = tumor, black = background)
        3. **MRI with the superimposed mask** (only in cases with a tumor)
        """
    )

    rows_dir = IMAGES_DIR

    # ------------------ CASOS CON TUMOR (row_*.png) ------------------
    tumor_rows = sorted(rows_dir.glob("tumor_*.png"))

    # ------------------ CASOS SIN TUMOR (example_no_tumor*.png) ------------------
    no_tumor_rows = sorted(rows_dir.glob("no_tumor*.png"))

    if not tumor_rows and not no_tumor_rows:
        st.error("Images Not found (0 and 1)")
        return

    # =========================
    # Contenedor central
    # =========================
    left_empty, center, right_empty = st.columns([1, 4, 1])
    with center:
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            tipo = st.radio("Show a case example: ",("🟢 Without tumor", "🔴 With tumor"),horizontal=True,index=0, width="stretch")

        if tipo == "🔴 With tumor":
            active_rows = tumor_rows
            state_key = "random_row_idx_tumor"
            boton_texto = "🔀 Show another tumor MRI image"
            titulo_prefix = "Cancer Patient"
            subtitulo_suffix = "Tumor RMI image"
        else:
            active_rows = no_tumor_rows
            state_key = "random_row_idx_no_tumor"
            boton_texto = "🔀 Show another healthy patient MRI image"
            titulo_prefix = "Healthy Patient"
            subtitulo_suffix = "No tumor RMI image"

        if not active_rows:
            if tipo == "🔴 With tumor":
                st.warning("Tumor images not found")
            else:
                st.warning("No tumor images not fount")
            return

        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        bc1, bc2, bc3 = st.columns([1, 2, 1])
        with bc2:
            if st.button(boton_texto,use_container_width=True):
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

        # =========================
        # Show images
        # =========================


        if tipo == "🔴 Con tumor":
            # fila row_XX con 3 columnas en una misma imagen
            img_row = Image.open(current_path)
            st.image(img_row, use_container_width=True)
            st.markdown(
                """
                <br></br>

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
                """,unsafe_allow_html=True)

        else:
            img_mri = Image.open(current_path).convert("RGB")
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(img_mri,  use_container_width=False)
            st.markdown(
                """
                <br></br>

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
                """,unsafe_allow_html=True)



def page_model():
    st.header("🧬 Deep learning model")
    st.markdown('<h5 style="text-align: center;color: black;"> <b> General Pipeline </b></h5>',unsafe_allow_html=True)
    st.image(IMAGES_DIR / "general_pipeline.png" ,use_container_width=True)
    st.markdown("## ResNet-50 Architecture")
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


def page_live_prediction():
    st.header("🔍 Live prediction with Flask model")

    st.markdown(
        """
        Upload an MRI image and the system will query the **deep learning model**
        deployed in Flask to predict whether there is a tumor or not.
        """
    )

    st.markdown(
        """
        In a realistic deployment, the input would often be an entire MRI study
        (many slices and sequences) rather than a single image. A more advanced system
        could:
        - Aggregate predictions across slices to provide a per-patient risk score.
        - Produce a 3D segmentation and estimate total tumor volume.
        - Track changes over time across multiple exams to quantify treatment response.
        Here we simplify this process to make the interaction easier to understand.
        """
    )

    st.sidebar.markdown("### ⚙️ Flask API configuration")
    api_url = st.sidebar.text_input("Base API URL", "http://localhost:8000")

    uploaded_file = st.file_uploader(
        "Upload an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )

    st.warning(
        """
        Never upload real patient-identifiable data to public demos.
        In real projects, DICOM images must be properly anonymized (removing names,
        IDs and any facial features) and handled under strict data protection and
        ethical guidelines.
        """
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded MRI", use_column_width=True)

        if st.button("Analyze MRI"):
            with st.spinner("Querying Flask model..."):
                try:
                    response = call_flask_model(api_url, pil_img)
                except Exception as e:
                    st.error(f"Error calling the API: {e}")
                    return

            st.markdown("### Model result")

            has_tumor = response.get("has_tumor", None)
            prob = response.get("probability", None)

            if has_tumor is None or prob is None:
                st.error(
                    "The API response does not contain the expected keys "
                    "(`has_tumor`, `probability`). Adapt the code to your format."
                )
            else:
                diagnosis = "TUMOR DETECTED" if has_tumor else "NO SIGNS OF TUMOR"
                color = "🔴" if has_tumor else "🟢"

                st.metric(label="Model diagnosis", value=f"{color} {diagnosis}")
                st.metric(label="Tumor probability", value=f"{prob * 100:.2f} %")

                st.markdown(
                    """
                    The reported probability should be interpreted as an approximate
                    **risk score**, not as a definitive diagnosis. Values close to 0.5
                    usually indicate uncertainty; in that range, the model should only
                    be used as a prompt for closer human review, never as an automatic
                    decision-maker.
                    """
                )

            mask_b64 = response.get("mask_base64", None)
            if mask_b64:
                st.markdown("### Segmentation mask (optional)")
                try:
                    mask_arr = decode_mask_from_b64(mask_b64)
                    st.image(
                        mask_arr,
                        caption="Mask predicted by the model",
                        use_column_width=True,
                    )
                except Exception:
                    st.info("The mask returned by the API could not be decoded.")

                st.caption(
                    "Segmentation masks allow automatic computation of tumor volume and shape "
                    "features (radiomics), which can be correlated with prognosis or molecular "
                    "subtypes in research studies."
                )


def page_media():
    st.header("🎥 Visual demo")

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


def page_collab():
    st.header("🤝 Collaboration")
    st.markdown("""

Collaboration is central to the success and scientific value of this brain tumor segmentation project. Our work builds directly on the collective efforts of the research community and the open-access initiatives that make high-quality medical imaging data available for machine learning research.

We acknowledge and thank the contributors of the **LGG MRI Segmentation** dataset, derived from the TCGA-LGG collection on *The Cancer Imaging Archive (TCIA)* and curated by Mateusz Buda. Their commitment to transparent data sharing enables researchers worldwide to develop, benchmark, and validate deep learning models for low-grade glioma segmentation. You can learn more about the dataset or contribute to their ongoing initiatives through the following links:
""")
    col1, col2, col3, col4, col5 = st.columns(
        [2, 5, 2, 5, 2], gap="large", vertical_alignment="center"
    )
    with col2:
        with st.container(
            border=True,
        ):
            st.image(str(KAGGLE_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        Kaggle – [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

                        </center>
                        """,
                unsafe_allow_html=True,
            )
    with col4:
        with st.container(border=True):
            st.image(str(TCIA_IMAGE), use_container_width=True)
            st.markdown(
                """
                        <center>

                        TCIA – [TCGA-LGG Collection](https://www.cancerimagingarchive.net)

                        </center>
                        """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """We also actively encourage collaboration within our own project. Our repository is publicly available, and we invite contributions related to model development, preprocessing pipelines, evaluation metrics, or exploratory radiogenomic analysis. Whether you are a researcher, clinician, or data scientist, your expertise can help improve the robustness and clinical relevance of our neural network models."""
    )

    col1, col2, col3, col4, col5 = st.columns(
        [2, 2, 5, 2, 2], gap="large", vertical_alignment="center"
    )
    with col3:
        with st.container(border=True):
            st.image(str(GITHUB_IMAGE))
            st.markdown(
                """
                        <center>

                        GitHub Repository [Brain Tumor Detection Project](https://github.com/FabsGMartin/brain-tumor-detection)

                        </center>
                        """,
                unsafe_allow_html=True,
            )
    st.markdown("""
We welcome pull requests, issue reporting, dataset discussions, and architectural improvements. In the spirit of open science, our goal is to create a collaborative space where insights and methods can be shared, replicated, and expanded. Through joint effort with both external data providers and the broader scientific community, we aim to produce reliable and reproducible tools that support research and clinical innovation in brain tumor analysis.
""")

    st.markdown(
        """
    ##### Support Cancer Research

    Beyond contributing to this project, you can also support the broader fight against cancer. Advancing treatments, improving diagnostics, and understanding tumor biology all depend on continued scientific and clinical research. Many organizations work tirelessly to fund studies, support patients, and accelerate the development of life-saving therapies.

    Here are several well-regarded associations you can collaborate with or donate to:

    - **American Cancer Society (ACS):** https://www.cancer.org
    - **Brain Tumor Foundation:** https://www.braintumorfoundation.org
    - **National Brain Tumor Society (NBTS):** https://braintumor.org
    - **Cancer Research UK:** https://www.cancerresearchuk.org
    - **European Organisation for Research and Treatment of Cancer (EORTC):** https://www.eortc.org

    Your support (whether through scientific collaboration, sharing expertise, or contributing to research foundations) helps move the field forward and brings us closer to better outcomes for patients around the world.
    """,
        unsafe_allow_html=True,
    )


def page_team():
    st.header("👥 Project team")

    st.markdown(
        """
        This work has been developed by a multidisciplinary team of data scientist with knowledge in  AIops.

        Below you can see our profiles and GitHub links.
        """
    )

    team = [
        {
            "name": "Luna Pérez T.",
            "github": "https://github.com/LunaPerezT",
            "linkedin": "https://www.linkedin.com/in/luna-p%C3%A9rez-troncoso-0ab21929b/",
        },
        {
            "name": "Raquel Hernández",
            "github": "https://github.com/RaquelH18",
            "linkedin": "https://www.linkedin.com/in/raquel-hern%C3%A1ndez-lozano/",
        },
        {
            "name": "Mary Marín",
            "github": "https://github.com/mmarin3011-cloud",
            "linkedin": "https://www.linkedin.com/in/mmarin30/",
        },
        {
            "name": "Fabián G. Martín",
            "github": "https://github.com/FabsGMartin",
            "linkedin": "",
        },
        {
            "name": "Miguel J. de la Torre",
            "github": "https://github.com/migueljdlt",
            "linkedin": "https://www.linkedin.com/in/miguel-jimenez-7403a2374/",
        },
        {
            "name": "Alejandro C.",
            "github": "https://github.com/alc98",
            "linkedin": "https://www.linkedin.com/in/alejandro-c-9b6525292/",
        },
    ]

    # Grid de 2 filas x 3 columnas, con GitHub justo debajo del nombre
    for row_start in range(0, len(team), 3):
        cols = st.columns(3)
        for col, member in zip(cols, team[row_start : row_start + 3]):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<h5 style="text-align: center;color: black;"> <b>{member["name"]}</b></h5>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                                <center>

                                **GitHub:** [{member["github"]}]({member["github"]})

                                </center>
                                """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                                <center>

                                **LinkedIn:** [{member["linkedin"]}]({member["linkedin"]})

                                </center>
                                """,
                        unsafe_allow_html=True,
                    )

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
    "🏠 Home",
    "📚 Introduction",
    "📂 Data Sources",
    "📊 Dataset Visualization",
    "🧠 MRI Images Visualization",
    "🧬 Deep learning model",
    "🔍 Live prediction",
    "🎥 Visual demo",
    "🤝 Collaboration",
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
elif choice == "📊 Dataset Visualization":
    page_data()
elif choice == "🧠 MRI Images Visualization":
    page_cases()
elif choice == "🧬 Deep learning model":
    page_model()
elif choice == "🔍 Live prediction":
    page_live_prediction()
elif choice == "🎥 Visual demo":
    page_media()
elif choice == "🤝 Collaboration":
    page_collab()
elif choice == "👥 About the Authors":
    page_team()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 1em;'>© 2025 Brain MRI Tumor Detection </p>",
    unsafe_allow_html=True,
)
