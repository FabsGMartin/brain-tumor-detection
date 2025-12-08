#-----------LIBRARIES LOADING-------------

import streamlit as st  
import pandas as pd
import plotly.express as px
from PIL import Image
import datetime
import requests
import base64
import io
import numpy as np

# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="Brain MRI Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    """
  ,unsafe_allow_html=True)

#-----------DATAFRAME AND MODEL LOADING-------------

df_routes_label = pd.read_csv("../data/routes_label.csv")
df_tumors =  pd.read_csv("../data/segmentation_routes_labels.csv")

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


try:
    from flask_app import app as backup_flask_app  # ajusta flask_app al nombre real
except Exception:
    backup_flask_app = None


def call_flask_model_backup(pil_image: Image.Image):
    if backup_flask_app is None:
        raise RuntimeError("Backup Flask app (.app) is not available or could not be imported.")

    pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    with backup_flask_app.test_client() as client:
        resp = client.post(
            "/predict",
            json={"image_base64": img_b64}
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Backup Flask app (.app) returned status {resp.status_code}: {resp.data!r}"
            )

        data = resp.get_json()
        if data is None:
            raise RuntimeError("Backup Flask app (.app) did not return valid JSON.")
        return data

def decode_mask_from_b64(mask_b64: str) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    return np.array(mask_img)

def page_home():


def page_intro():
    st.header("🧠 Brain tumor detection and segmentation")
    
    try:
        mri_img = Image.open("images/TCGA_CS_4942_19970222_10.tif")
        st.image(
            mri_img,
            caption="Example brain MRI (TCGA_CS_4942_19970222_10)",
            use_column_width=True
        )
    except Exception:
        st.info(
            "Place the MRI image at `images/TCGA_CS_4942_19970222_10.tif` "
            "or update the path in `page_intro()`."
        )
    
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

<<<<<<< HEAD

def page_sources():
    '''
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

    '''

def page_dataset():
    st.header("📊 Database analysis")
=======
>>>>>>> c99056271959a403e1adb1171ea643a6bd6f19be


def page_cases():
    st.header("🖼️ Example cases: negative vs positive")

    st.markdown(
        "In this section we show an example patient **without tumor** (negative case) "
        "and a patient **with tumor** (positive case), together with their segmentation masks."
    )

    st.markdown(
        """
        In daily practice, neuroradiologists do not only answer “tumor yes/no”.
        They carefully assess:
        - **Location** of the lesion (e.g. frontal vs temporal lobe) and involvement of
          eloquent areas (motor, language).
        - **Mass effect and midline shift**, which indicate how much the lesion compresses
          surrounding structures.
        - **Edema and infiltration** patterns, often best seen on FLAIR.
        - **Contrast enhancement** behavior, which can suggest high-grade components
          or breakdown of the blood-brain barrier.
        These qualitative descriptors, combined with clinical data, guide biopsy, surgery
        and follow-up strategy.
        """
    )

    neg_img_path = "images/caso_negativo_mri.png"
    neg_mask_path = "images/caso_negativo_mask.png"
    pos_img_path = "images/caso_positivo_mri.png"
    pos_mask_path = "images/caso_positivo_mask.png"


    st.markdown("### Negative case (no tumor)")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("MRI – negative case")
        try:
            neg_img = Image.open(neg_img_path)
            st.image(neg_img, use_column_width=True)
        except Exception:
            st.info(f"Place the negative case image at `{neg_img_path}`.")

    with col2:
        st.caption("Mask – negative case (no tumor)")
        try:
            neg_mask = Image.open(neg_mask_path)
            st.image(neg_mask, use_column_width=True)
        except Exception:
            st.info(f"Place the negative case mask at `{neg_mask_path}`.")

    st.markdown("---")
    st.markdown("### Positive case (with tumor)")
    col3, col4 = st.columns(2)

    with col3:
        st.caption("MRI – positive case")
        try:
            pos_img = Image.open(pos_img_path)
            st.image(pos_img, use_column_width=True)
        except Exception:
            st.info(f"Place the positive case image at `{pos_img_path}`.")

    with col4:
        st.caption("Mask – positive case (tumor in red)")
        try:
            pos_mask = Image.open(pos_mask_path)
            st.image(pos_mask, use_column_width=True)
        except Exception:
            st.info(f"Place the positive case mask at `{pos_mask_path}`.")

    st.info(
        """
        The visual examples here are based on single 2D slices. Real-world decisions
        are based on full 3D studies and multiple MRI sequences, as well as clinical
        information. The goal of this demo is educational: to illustrate how a model
        can highlight suspicious regions and complement expert image interpretation.
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
        "Upload an MRI image (PNG/JPG)",
        type=["png", "jpg", "jpeg"]
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
                    used_backup = False
                except Exception as e_main:
                    st.error(f"Error calling main Flask API ({api_url}): {e_main}")
                    st.info("Trying backup local Flask app (.app) instead...")

             
                    try:
                        response = call_flask_model_backup(pil_img)
                        used_backup = True
                    except Exception as e_backup:
                        st.error(f"Backup Flask app (.app) also failed: {e_backup}")
                        return

            st.markdown("### Model result")

            if 'used_backup' in locals() and used_backup:
                st.caption("Result obtained from backup Flask app (.app).")

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

                st.metric(
                    label="Model diagnosis",
                    value=f"{color} {diagnosis}"
                )
                st.metric(
                    label="Tumor probability",
                    value=f"{prob*100:.2f} %"
                )

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
                    st.image(mask_arr, caption="Mask predicted by the model", use_column_width=True)
                except Exception:
                    st.info("The mask returned by the API could not be decoded.")

                st.caption(
                    "Segmentation masks allow automatic computation of tumor volume and shape "
                    "features (radiomics), which can be correlated with prognosis or molecular "
                    "subtypes in research studies."
                )


def page_media():
    st.header("🎥 Visual demo and appointment")
    
    st.subheader("Demo video of the app / model")
    try:
        with open("video.mp4", "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception:
        st.info("Place a `video.mp4` file next to `app.py` or update the path in the code.")

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

def page_contribute():
    st.header("🤝 Contribute and support patients")

    # Mensaje en verde
    st.success("June 8, 2023 – International Brain Tumor Day\n\n")

    st.markdown(
        """
        If you would like to support patients and research related to brain tumors
        and cancer in general, you can contribute through the  
        **Asociación Española Contra el Cáncer (AECC)**.
        
        The AECC is a non-profit organization that has been working for decades across
        Spain to **reduce the impact of cancer on society**. Its activity is based on
        three main pillars:
        
        - **Support for patients and families**: emotional, psychological and social care,
          guidance on practical aspects (work, benefits, legal issues) and accompaniment
          throughout the disease.
        - **Prevention and health education**: awareness campaigns about risk factors,
          promotion of healthy lifestyles and early detection programs.
        - **Funding of research**: competitive grants to research groups in oncology to
          improve diagnosis, treatment and survival.
        """
    )

    st.markdown(
        """
        In practice, the AECC acts through a network of local offices, hospitals and
        volunteer programs. They offer:
        
        - Free **psycho-oncological support** for patients and relatives.
        - **Social workers** who help manage administrative procedures and resources.
        - **Telephone and online helplines** to answer questions and provide guidance.
        - Collaboration with hospitals and research centers to drive **clinical and
          translational research**.
        """
    )

  
    st.subheader("📅 Follow-up appointment")

    cita = st.date_input(
        "Select a date for the follow-up visit",
        datetime.date.today(),
        key="contribute_date"
    )

    # Logo clicable de la AECC dentro del bloque de cita
    aecc_logo_url = "https://www.aecc.es/sites/default/files/styles/ps_xl/public/logo-aecc.png"
    st.markdown(
        f"""
        <a href="https://www.aecc.es/" target="_blank">
            <img src="{aecc_logo_url}" alt="AECC" width="260">
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        By clicking on the logo, you will be redirected to the official website of the
        Spanish Association Against Cancer, where you can:
        - Request information and support for you or your family.
        - Collaborate as a donor or volunteer.
        - Learn more about prevention and early detection.
        - Help fund cutting-edge cancer research projects.
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
            "photo": "images/team/miembro1.jpg"
        },
        {
            "name": "Raquel Hernández",
            "github": "https://github.com/RaquelH18",
            "photo": "images/team/miembro2.jpg"
        },
        {
            "name": "María  Marín",
            "github": "https://github.com/mmarin3011-cloud",
            "photo": "images/team/miembro3.jpg"
        },
        {
            "name": "Fabián G. Martín",
            "github": "https://github.com/FabsGMartin",
            "photo": "images/team/miembro4.jpg"
        },
        {
            "name": "Miguel J. de la Torre",
            "github": "https://github.com/migueljdlt",
            "photo": "images/team/miembro5.jpg"
        },
        {
            "name": "Alejandro C.",
            "github": "https://github.com/alc98",
            "photo": "images/team/miembro6.jpg"
        },
    ]

   
    for row_start in range(0, len(team), 3):
        cols = st.columns(3)
        for col, member in zip(cols, team[row_start:row_start + 3]):
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


st.markdown('''<h1 style="text-align: center;color: black;"> <b> Brain MRI Tumor Detection </b></h1>
    <h5 style="text-align: center;color: gray"> <em> A Deep Learning based project to detect and segmetate brain tumors in MRI images </em> </h5>''', unsafe_allow_html=True)
st.markdown("---")

# ---------- SIDEBAR NAVIGATION ----------

st.sidebar.header("Navigation Menu")
st.sidebar.caption("Choose a section to explore the project.")

menu = [
        "🏠 Home"
        "📚 Introduction",
        "📂 Data Sources",
        "🧬 Deep learning model",
<<<<<<< HEAD
        "📊 Data Visualization",
=======
>>>>>>> c99056271959a403e1adb1171ea643a6bd6f19be
        "🖼️ Example cases",
        "🔍 Live prediction",
        "🎥 Media and appointment",
        "🤝 Contribute",
        "👥 About the Authors"
    ]

choice = st.sidebar.radio("Select a page:", menu)

# ---------- APP BODY ----------

if choice == "🏠 Home":
    page_home()
elif choice == "📚 Introduction":
    page_intro()
elif choice ==  "📂 Data Sources":  
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
elif choice == "🤝 Contribute":
    page_contribute()
elif choice == "👥 Team":
    page_team()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 1em;'>© 2025 Brain MRI Tumor Detection </p>",unsafe_allow_html=True)




















