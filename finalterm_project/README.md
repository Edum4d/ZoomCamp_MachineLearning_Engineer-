# Midterm Project – Loan Repayment Prediction (Neural Networks)

This repository contains my **Midterm Project** for the **Machine Learning Zoomcamp**, implemented using a **Deep Neural Network (Keras / TensorFlow)**.

The project is based on the **Kaggle Playground Series 2025** competition, where the objective is to predict the probability that a borrower will **pay back their loan**.  
It covers the **full ML lifecycle**:

- Exploratory Data Analysis (EDA)
- Feature engineering & encoding
- Neural Network training and tuning
- Model selection (AUC / F1)
- Training a final model on the full dataset
- Serving the model via a **FastAPI** API

---

## 📌 Problem Statement

Given information about a borrower and their loan application, we want to estimate:

> **What is the probability that this borrower will pay back their loan?**

Target variable:

- `loan_paid_back`
  - `1` → loan was paid back  
  - `0` → loan was not paid back (default)

---

## 🧾 Dataset

Data comes from the **Kaggle Playground Series 2025** competition.

Features include:

### Numerical
- `annual_income`
- `debt_to_income_ratio`
- `loan_amount`

### Categorical
- `grade_subgrade`
- `loan_purpose`
- `employment_status`
- `education_level`
- `gender`
- `marital_status`

### Engineered
- `loan_to_income`
- `total_debt_est`
- `loan_to_total_debt`

---

## 🗂 Project Structure

```text
.
├── train.csv
├── notebook.ipynb
├── train_nn.py
├── app.py
├── nn_model_final.keras
├── nn_best_params.json
├── scaler_nn.pkl
├── one_hot_encoder.pkl
├── ordinal_encoder.pkl
├── nn_feature_columns.json
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Model Development

The notebook includes:

- EDA
- Feature engineering
- Neural Network experiments
- Hyperparameter tuning
- Model selection based on **ROC-AUC** and **F1-score**

The best configuration is saved to:

```text
nn_best_params.json
```

---

## 🏋️ Train Final Model

```bash
python train_nn.py
```

This trains the final Neural Network on the **full dataset** and saves:

- Model
- Scaler
- Encoders
- Feature column order

---

## 🌐 FastAPI Inference Server

Start the API:

```bash
uvicorn app:app --reload
```

Endpoints:

- `GET /` – health check
- `POST /predict` – loan default prediction

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 📥 Example Request

```json
{
  "grade_subgrade": "D5",
  "loan_purpose": "Other",
  "employment_status": "Employed",
  "education_level": "High School",
  "gender": "Female",
  "marital_status": "Single",
  "loan_amount": 11461.42,
  "annual_income": 28781.05,
  "debt_to_income_ratio": 0.049
}
```

---

## 📤 Example Response

```json
{
  "prob_default": 0.27,
  "threshold": 0.5,
  "prediction": 0
}
```

---

## 🔁 End-to-End Workflow

1. Explore and tune models in `notebook.ipynb`
2. Train final model with `train_nn.py`
3. Start API with `uvicorn app:app --reload`
4. Send requests to `/predict`

---

## 🧠 Notes

- Feature order is enforced via `nn_feature_columns.json`
- Scaling is mandatory for neural networks
- All preprocessing steps are reusable and deterministic
