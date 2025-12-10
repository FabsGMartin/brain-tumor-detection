# Brain Tumor Detection & Segmentation 🧠

A comprehensive pipeline for detecting and segmenting brain tumors (especially low-grade gliomas) from MRI scans. The system combines deep learning models, data preprocessing, and a web interface for easy deployment and use.  

## 🚀 Project Overview

We developed a two-stage deep learning pipeline:

- **Stage 1 – Classification:** A pre-trained **ResNet-50** model filters MRI slices to detect which contain a tumor (vs. healthy slices).  
- **Stage 2 – Segmentation:** A **ResUNet**-based model performs pixel-level segmentation on the slices identified as containing a tumor — delineating the tumor region.  
- The pipeline is supported by data preprocessing, augmentation, and evaluation metrics to ensure robustness and reliability.  
- Deployment via a **Flask backend + Streamlit frontend** transforms the pipeline into a user-friendly diagnostic tool accessible for clinicians and researchers.

## 📁 Repository Structure
|
├─ data/ ← raw and processed MRI routes and mask results
├─ models/ ← trained model weights and saved artifacts
├─ notebooks/ ← Jupyter notebooks for data exploration, training, evaluation
├─ backend-flask/ ← backend API for inference
├─ frontend-streamlit/ ← Streamlit app for visualization & live prediction
└─ README.md ← this file


## 🔧 How to Use

1. Prepare your MRI dataset under `data/`.  
2. (Optional) Run the notebooks for data preprocessing, augmentation, and training.  
3. Load a trained model from `models/`.  
4. Start the backend API (Flask).  
5. Launch the frontend (Streamlit) for uploading MRI slices and visualizing predictions + segmentation masks.

## 📊 Results & Impact

Thanks to the two-stage approach:

- Classification allows efficient filtering of healthy images, saving computational resources.  
- Segmentation produces precise masks that highlight tumor regions — potentially useful for diagnosis, follow-up, or radiomic studies.  
- The tool is designed to accelerate analysis, reduce human error, and support early detection — which is critical for patient prognosis in low-grade gliomas.  

## 🤝 Collaboration & Contribution

This is an open project. We welcome collaborators — from data scientists to clinicians — who want to:  

- Expand the dataset,  
- Explore alternative architectures or loss functions,  
- Integrate more sequences/modalities, or  
- Help improve the interface and usability.  

If you contribute, please fork the repo, create a feature branch, and propose pull requests.  

## 📝 License

This project is distributed under the MIT License.  
