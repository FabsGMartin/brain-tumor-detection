# Brain Tumor Detection & Segmentation 🧠

A comprehensive pipeline for detecting and segmenting brain tumors (especially low-grade gliomas) from MRI scans. The system combines deep learning models, data preprocessing, and a web interface for easy deployment and use.

## 🚀 Project Overview

We developed a two-stage deep learning pipeline:

- **Stage 1 – Classification:** A pre-trained **ResNet-50** model filters MRI slices to detect which contain a tumor (vs. healthy slices).
- **Stage 2 – Segmentation:** A **ResUNet**-based model performs pixel-level segmentation on the slices identified as containing a tumor — delineating the tumor region.
- The pipeline is supported by data preprocessing, augmentation, and evaluation metrics to ensure robustness and reliability.
- Deployment via a **Flask backend + Streamlit frontend** transforms the pipeline into a user-friendly diagnostic tool accessible for clinicians and researchers.

## 📁 Repository Structure

```
├─ data/                    # Raw and processed MRI routes and mask results
├─ src/
│  ├─ backend/             # Flask API backend
│  │  ├─ app.py           # Main Flask application
│  │  ├─ model.py         # Model loading and S3 integration
│  │  ├─ storage.py       # S3 prediction storage management
│  │  └─ models/          # Local model cache (models loaded from S3)
│  └─ frontend/           # Streamlit frontend
│     ├─ ui.py            # Main Streamlit application
│     ├─ img/             # Static images
│     └─ static/          # Static assets
├─ notebooks/             # Jupyter notebooks for data exploration, training, evaluation
├─ requirements-backend.txt    # Backend dependencies
├─ requirements-frontend.txt   # Frontend dependencies
├─ Dockerfile             # Dockerfile for local development
├─ Dockerfile.backend     # Optimized Dockerfile for AWS App Runner
├─ docker-compose.yml     # Docker Compose for local development
├─ apprunner.yaml         # AWS App Runner configuration
├─ .env                   # Environment variables template
└─ README.md              # This file
```

## 🏗️ Architecture

The project uses a cloud-native architecture:

- **Backend (AWS App Runner):** Flask API serving deep learning models, loading models and data from S3
- **Frontend (Streamlit Community Cloud):** Interactive web interface connecting to the backend API
- **Storage (AWS S3):** All data stored in S3 buckets:
  - Models for inference
  - Data files (CSVs)
  - Prediction results and history (JSON files)
  - Segmentation masks (PNG files)

## 🔧 Local Development

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- AWS account with S3 buckets configured (optional for local development)

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd brain-tumor-detection
   ```

2. **Configure environment file:**

   ```bash
   # Edit .env with your configuration
   # The .env file is already included in the repository as a template
   ```

3. **Configure environment variables:**

   - Set up AWS credentials if using S3
   - Configure S3 bucket names for models and data
   - Set API URLs and CORS origins

4. **Run with Docker Compose:**

   ```bash
   docker-compose up --build
   ```

   This will start:

   - Backend API at `http://localhost:5000`
   - Frontend UI at `http://localhost:8501`

5. **Or run manually:**

   **Backend:**

   ```bash
   pip install -r requirements-backend.txt
   python src/backend/app.py
   ```

   **Frontend:**

   ```bash
   pip install -r requirements-frontend.txt
   streamlit run src/frontend/ui.py
   ```

## ☁️ Cloud Deployment

### Backend: AWS App Runner

1. **Prepare S3 Buckets:**

   - Create a bucket for models (e.g., `your-models-bucket`)
   - Create a bucket for data (e.g., `your-data-bucket`)
   - Upload models to `models/` prefix in the models bucket:
     - `classifier-resnet-model-final.keras`
     - `segmentation_ResUNet_final.keras`
   - Upload CSV files to `data/` prefix in the data bucket:
     - `route_label.csv`
     - `segmentation_routes_labels.csv`

2. **Configure AWS App Runner:**

   - Go to AWS App Runner console
   - Create a new service
   - Choose "Source code repository" or "Container registry"
   - Use the `Dockerfile.backend` for building
   - Configure environment variables in App Runner:
     - `AWS_ACCESS_KEY_ID`
     - `AWS_SECRET_ACCESS_KEY`
     - `AWS_DEFAULT_REGION`
     - `S3_MODELS_BUCKET`
     - `S3_DATA_BUCKET`
     - `CORS_ORIGINS` (your Streamlit Cloud URL)
     - Other variables from `.env`

3. **Deploy:**
   - App Runner will automatically build and deploy from your repository
   - Note the service URL (e.g., `https://xxxxx.us-east-1.awsapprunner.com`)

### Frontend: Streamlit Community Cloud

1. **Push to GitHub:**

   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Configure Streamlit Cloud:**

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set:
     - **Main file path:** `src/frontend/ui.py`
     - **Python version:** 3.9
   - Add secrets in Streamlit Cloud settings (Settings → Secrets):
     ```toml
     AWS_ACCESS_KEY_ID = "your_access_key"
     AWS_SECRET_ACCESS_KEY = "your_secret_key"
     AWS_DEFAULT_REGION = "us-east-1"
     S3_DATA_BUCKET = "your-data-bucket"
     S3_DATA_PREFIX = "data/"
     API_URL = "https://your-app-runner-url.us-east-1.awsapprunner.com"
     ```

3. **Deploy:**
   - Click "Deploy"
   - Streamlit Cloud will install dependencies from `requirements-frontend.txt`
   - Your app will be available at `https://your-app.streamlit.app`

### Environment Variables Reference

See `.env` for a complete list. Key variables:

- **AWS Configuration:**

  - `AWS_ACCESS_KEY_ID`: AWS access key
  - `AWS_SECRET_ACCESS_KEY`: AWS secret key
  - `AWS_DEFAULT_REGION`: AWS region (e.g., `us-east-1`)

- **S3 Configuration:**

  - `S3_MODELS_BUCKET`: Bucket name for storing models
  - `S3_DATA_BUCKET`: Bucket name for storing data files and predictions
  - `S3_MODELS_PREFIX`: Prefix for models (default: `models/`)
  - `S3_DATA_PREFIX`: Prefix for data files (default: `data/`)
  - `S3_PREDICTIONS_PREFIX`: Prefix for predictions storage (default: `predictions/`)

- **Backend Configuration:**

  - `PORT`: Server port (default: `5000`)
  - `FLASK_ENV`: Environment (`production` or `development`)
  - `CORS_ORIGINS`: Comma-separated list of allowed origins

- **Frontend Configuration:**
  - `API_URL`: Backend API URL

## 📊 Results & Impact

Thanks to the two-stage approach:

- Classification allows efficient filtering of healthy images, saving computational resources.
- Segmentation produces precise masks that highlight tumor regions — potentially useful for diagnosis, follow-up, or radiomic studies.
- The tool is designed to accelerate analysis, reduce human error, and support early detection — which is critical for patient prognosis in low-grade gliomas.

## 🔒 Security Notes

- **Never commit `.env` files** - They contain sensitive credentials
- Use AWS IAM roles with minimal permissions in production
- Configure CORS origins to only allow your Streamlit Cloud domain
- Validate and sanitize all user inputs
- Use HTTPS in production (enforced by App Runner and Streamlit Cloud)

## 🐛 Troubleshooting

### Models not loading

- Verify S3 bucket names and prefixes are correct
- Check AWS credentials have read permissions for S3
- Verify model files exist in S3 with correct names

### CORS errors

- Ensure `CORS_ORIGINS` includes your Streamlit Cloud URL
- Check backend logs for CORS-related messages

### API connection errors

- Verify `API_URL` in frontend matches your App Runner service URL
- Check App Runner service is running and healthy
- Verify `/health` endpoint returns 200

## 🤝 Collaboration & Contribution

This is an open project. We welcome collaborators — from data scientists to clinicians — who want to:

- Expand the dataset,
- Explore alternative architectures or loss functions,
- Integrate more sequences/modalities, or
- Help improve the interface and usability.

If you contribute, please fork the repo, create a feature branch, and propose pull requests.

## 📝 License

This project is distributed under the MIT License.
