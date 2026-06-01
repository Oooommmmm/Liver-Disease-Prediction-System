# Liver Disease Prediction System
 
A machine learning web application that predicts the likelihood of liver disease based on patient clinical parameters — built with Python, Scikit-learn, and Streamlit.
 
 
## Features
 
- Predicts liver disease risk based on 10 clinical input parameters
- Trained on the **Indian Liver Patient Dataset (ILPD)** — 583 real patients
- Displays **confidence score (%)** alongside the prediction
- Clean two-column UI with dataset insights and KPI cards
- Real-time prediction with color-coded results (🔴 High Risk / ✅ Low Risk)
---
 
## Model Details
 
| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | Indian Liver Patient Dataset (ILPD) |
| Total Patients | 583 |
| Disease Cases | 416 (71.3%) |
| Train/Test Split | 80% / 20% |
| **Model Accuracy** | **75%** |
 
> **Note:** The ILPD dataset has a 71% disease prevalence rate which introduces class imbalance. Predictions reflect dataset distribution and should not be used as standalone clinical decisions.
 
---
 
## Input Parameters
 
| Parameter | Description |
|---|---|
| Age | Age of the patient |
| Gender | Male / Female |
| Total Bilirubin | Bilirubin level in blood |
| Direct Bilirubin | Direct bilirubin level |
| Alkaline Phosphotase | Enzyme level (liver function indicator) |
| SGPT (Alamine Aminotransferase) | Liver enzyme level |
| SGOT (Aspartate Aminotransferase) | Liver enzyme level |
| Total Proteins | Total protein level in blood |
| Albumin | Albumin protein level |
| Albumin/Globulin Ratio | Ratio of albumin to globulin |
 
---
 
## How to Run Locally
 
**1. Clone the repository**
```bash
git clone https://github.com/your-username/liver-disease-prediction.git
cd liver-disease-prediction
```
 
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
 
**3. Train the model (generates liver_model.pkl)**
```bash
python train_model.py
```
 
**4. Run the Streamlit app**
```bash
streamlit run app.py
```
 
---
 
 
**Disclaimer:** This tool is built for educational purposes only and is not a substitute for professional medical advice.
 

