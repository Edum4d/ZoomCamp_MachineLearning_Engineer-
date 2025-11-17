# Midterm Project – Loan Repayment Prediction (ML Zoomcamp)

This repository contains my **Midterm Project** for the **Machine Learning Zoomcamp**.

The project is based on the **Kaggle Playground Series 2025** competition, where the goal is to predict the probability that a borrower will **pay back their loan**. I use the competition dataset as the basis for a full ML pipeline:

- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Model selection and tuning
- Serving the model via an API (FastAPI)
- Containerizing the API with Docker

---

## 📌 Problem Statement

Given information about a borrower and their loan application, we want to estimate:

> **What is the probability that this borrower will pay back their loan?**

Formally, the competition’s target variable is:

- `loan_paid_back`
  - `1` → loan was paid back  
  - `0` → loan was not paid back (default)


---

## 🧾 Dataset

The data comes from the **Kaggle Playground Series 2025** competition.  
Each row represents one loan application with:

- **Borrower / Financial information**
  - `annual_income`
  - `debt_to_income_ratio`
  - `credit_score`

- **Loan information**
  - `loan_amount`
  - `interest_rate`
  - `loan_purpose`
  - `grade_subgrade`

- **Demographic / Profile information**
  - `gender`
  - `marital_status`
  - `education_level`
  - `employment_status`

Example row:

| id    | annual_income | debt_to_income_ratio | credit_score | loan_amount | interest_rate | gender | marital_status | education_level | employment_status | loan_purpose | grade_subgrade |
|-------|---------------|----------------------|--------------|------------:|---------------|--------|----------------|-----------------|-------------------|--------------|----------------|
| 593994 | 28781.05     | 0.049                | 626          | 11461.42    | 14.73         | Female | Single         | High School     | Employed          | Other        | D5             |


---

## 🗂 Project Structure

Example layout (may vary slightly):

```text
.
├── data/
│   ├── train.csv                # Kaggle training data
│   └── test.csv                 # Kaggle test data (no target)
├── EDA + Models tuning.ipynb    # EDA, model experiments, hyperparameter tuning
├── train.py                     # Training script (data prep, model training, saving model)
├── app.py                       # FastAPI app exposing /predict endpoint
├── xgb_model_final.json         # Trained XGBoost model (loaded by app.py)
├── dv_xgb.pkl                   # DictVectorizer (or similar) used for features
├── one_hot_encoder.pkl          # One-hot encoder for categorical features
├── ordinal_encoder.pkl          # Ordinal encoder (if used)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition for the API
└── README.md                    # This file
```

> **Note:** The exact filenames of the model artifacts match what is loaded in `app.py`.

---

## ⚙️ Environment & Installation (Local)

I use a **virtual environment** plus a **requirements file** for reproducibility.

### 1. Clone the repository

```bash
git clone <this-repo-url>
cd <this-repo-folder>
```

### 2. Create & activate virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scriptsctivate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Model Training

The training pipeline is implemented in `train.py`. It generally:

1. Loads the Kaggle training data from `data/train.csv`.
2. Splits it into **train/validation** sets.
3. Performs preprocessing and feature encoding.
4. Optionally applies **undersampling** to handle class imbalance.
5. Trains an XGBoost classification model to predict `loan_paid_back`.
6. Uses the **tuned hyperparameters** (found in the notebook) for the final model.
7. Saves the trained model and preprocessing artifacts to disk:

   - `xgb_model_final.json`
   - `dv_xgb.pkl`
   - `one_hot_encoder.pkl`
   - `ordinal_encoder.pkl`

Run:

```bash
python train.py
```

After this step, the model artifacts are ready for inference and for serving with the API.

---

## 🌐 Model Deployment (FastAPI API – Local)

The deployment is done using **FastAPI** in `app.py`.

### Start the API server

```bash
uvicorn app:app --reload
```

By default, the API is available at:

- Base URL: `http://127.0.0.1:8000`
- Prediction endpoint: `http://127.0.0.1:8000/predict`

### Request format

The `/predict` endpoint expects a JSON payload with the borrower and loan features (the same columns used during training, except for `loan_paid_back`).

Example request body:

```json
{
  "annual_income": 28781.05,
  "debt_to_income_ratio": 0.049,
  "credit_score": 626,
  "loan_amount": 11461.42,
  "interest_rate": 14.73,
  "gender": "Female",
  "marital_status": "Single",
  "education_level": "High School",
  "employment_status": "Employed",
  "loan_purpose": "Other",
  "grade_subgrade": "D5"
}
```

### Example `curl` request

```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{
       "annual_income": 28781.05,
       "debt_to_income_ratio": 0.049,
       "credit_score": 626,
       "loan_amount": 11461.42,
       "interest_rate": 14.73,
       "gender": "Female",
       "marital_status": "Single",
       "education_level": "High School",
       "employment_status": "Employed",
       "loan_purpose": "Other",
       "grade_subgrade": "D5"
     }'
```

### API response

The API (see `app.py`) returns:

- `prob_default` – predicted probability that the loan will **default** (`loan_paid_back = 0`)
- `threshold` – decision threshold used to convert probability into a class (currently 0.5)
- `prediction` – final binary prediction:
  - `1` → predicted **default** (loan not paid back)  
  - `0` → predicted **no default** (loan will be paid back)

---

## 🐳 Running with Docker

This project is containerized so you can run the FastAPI service without setting up a local Python environment.

### Prerequisites

- [Docker](https://www.docker.com/) installed
- The following files present in the project root:
  - `app.py`
  - `xgb_model_final.json`
  - `dv_xgb.pkl`
  - `one_hot_encoder.pkl`
  - `ordinal_encoder.pkl`
  - `requirements.txt`
  - `Dockerfile`

### Build the Docker image

From the project root, run:

```bash
docker build -t loan-default-api .
```

This command:

- Installs all Python dependencies from `requirements.txt`
- Copies the application and model artifacts into the image
- Exposes port `8000` for the API

### Run the container

```bash
docker run --rm -p 8000:8000 loan-default-api
```

The API will then be available at:

- Base URL: `http://127.0.0.1:8000`
- Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Prediction endpoint: `POST http://127.0.0.1:8000/predict`

Example `curl` request (same as local):

```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{
       "annual_income": 28781.05,
       "debt_to_income_ratio": 0.049,
       "credit_score": 626,
       "loan_amount": 11461.42,
       "interest_rate": 14.73,
       "gender": "Female",
       "marital_status": "Single",
       "education_level": "High School",
       "employment_status": "Employed",
       "loan_purpose": "Other",
       "grade_subgrade": "D5"
     }'
```

You should receive a JSON response containing `prob_default`, `threshold`, and `prediction`.

---

## 🔁 How to Reproduce the Project End-to-End

This section describes how to reproduce the whole pipeline:
from raw data → trained model → running API.

### Step 1 – Get the data

1. Go to the **Kaggle Playground Series 2025** competition page.
2. Download the dataset (`train.csv`, `test.csv`).
3. Place the files under a `data/` folder in the project root:

   ```text
   data/
   ├── train.csv
   └── test.csv
   ```

### Step 2 – Create environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate       # or venv\Scriptsctivate on Windows
pip install -r requirements.txt
```

### Step 3 – Explore & select a model (optional but recommended)

1. Open `EDA + Models tuning.ipynb` in Jupyter or VS Code.
2. Run the notebook:
   - Perform EDA.
   - Try different models and hyperparameters.
   - Decide on the final model configuration (e.g. XGBoost + specific params).
3. Update `train.py` if needed to reflect your chosen configuration.

### Step 4 – Train the final model

Run:

```bash
python train.py
```

This will:

- Load `data/train.csv`
- Preprocess and encode features
- Train the final XGBoost model
- Save the artifacts:

  - `xgb_model_final.json`
  - `dv_xgb.pkl`
  - `one_hot_encoder.pkl`
  - `ordinal_encoder.pkl`

### Step 5 – Serve the model via FastAPI (local)

```bash
uvicorn app:app --reload
```

Then:

- Open the docs at `http://127.0.0.1:8000/docs`
- Or send a POST request to `http://127.0.0.1:8000/predict` with borrower/loan data.

### Step 6 – Run the model in Docker (optional but recommended)

1. Make sure the model artifacts from Step 4 are present in the project root.
2. Build the image:

   ```bash
   docker build -t loan-default-api .
   ```

3. Run the container:

   ```bash
   docker run --rm -p 8000:8000 loan-default-api
   ```

4. Interact with the API at `http://127.0.0.1:8000` as before.


---

## 🧠 How the Code Fits Together (Summary)

1. **EDA + Models tuning notebook (`EDA + Models tuning.ipynb`)**
   - Used for **Exploratory Data Analysis** and model experimentation.
   - From these experiments, I select the **best-performing model configuration**.

2. **Training script (`train.py`)**
   - Implements the **final training pipeline** using the chosen configuration.
   - Loads data, preprocesses features, handles imbalance, trains XGBoost, and saves artifacts (`xgb_model_final.json`, encoders).

3. **Server script (`app.py`)**
   - Implements a **FastAPI** application.
   - On startup, it **loads the trained model and preprocessing pipeline** from disk.
   - Exposes a `/predict` endpoint that:
     - accepts JSON with the same features used during training,
     - applies the same preprocessing steps,
     - uses the model to compute the probability of default,
     - applies a threshold (0.5) to produce a binary prediction,
     - returns the result as JSON.
