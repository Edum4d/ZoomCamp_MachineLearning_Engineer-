from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import json

import tensorflow as tf
from tensorflow import keras

# -----------------------
# Carregar artefactos
# -----------------------
model = keras.models.load_model("nn_model_final.keras")

with open("scaler_nn.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("one_hot_encoder.pkl", "rb") as f:
    one_hot = pickle.load(f)

with open("ordinal_encoder.pkl", "rb") as f:
    ordinal = pickle.load(f)

# Ordem final das colunas (IMPORTANTÍSSIMO para NN + scaler)
with open("nn_feature_columns.json", "r") as f:
    FEATURE_COLS = json.load(f)

# Mesmas listas de colunas que usaste no treino
ordinal_cols = ["grade_subgrade", "loan_purpose", "employment_status", "education_level"]
one_hot_cols = ["gender", "marital_status"]

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Loan Default Prediction API (Neural Net)")

# -----------------------
# Esquema de entrada
# (adapta / adiciona campos numéricos que existam no teu df)
# -----------------------
class LoanRow(BaseModel):
    # categóricas
    grade_subgrade: str
    loan_purpose: str
    employment_status: str
    education_level: str
    gender: str
    marital_status: str

    # numéricas básicas
    loan_amount: float
    annual_income: float
    debt_to_income_ratio: float

    # se tiveres mais features numéricas no df original, adiciona aqui:
    # credit_score: float
    # age: float


@app.get("/")
def root():
    return {"message": "Loan prediction API (NN) up!"}


@app.post("/predict")
def predict(row: LoanRow):
    # -----------------------
    # 1) Converter input em DataFrame
    # -----------------------
    df = pd.DataFrame([row.dict()])

    # -----------------------
    # 2) Aplicar os encoders carregados
    # -----------------------
    # One-hot
    one_hot_arr = one_hot.transform(df[one_hot_cols])
    one_hot_df = pd.DataFrame(
        one_hot_arr,
        columns=one_hot.get_feature_names_out(one_hot_cols),
        index=df.index,
    )

    # Ordinal
    ordinal_arr = ordinal.transform(df[ordinal_cols])
    ordinal_df = pd.DataFrame(
        ordinal_arr,
        columns=ordinal_cols,
        index=df.index,
    )

    # Juntar tudo: drop originais categóricas, adicionar encodadas
    df_encoded = pd.concat(
        [
            df.drop(one_hot_cols + ordinal_cols, axis=1),
            ordinal_df,
            one_hot_df,
        ],
        axis=1,
    )

    # -----------------------
    # 3) Features derivadas (mesmo código que no treino)
    # -----------------------
    df_encoded["loan_to_income"] = df_encoded["loan_amount"] / (df_encoded["annual_income"] + 1e-6)
    df_encoded["total_debt_est"] = (df_encoded["annual_income"] * df_encoded["debt_to_income_ratio"]) / 100
    df_encoded["loan_to_total_debt"] = df_encoded["loan_amount"] / (df_encoded["total_debt_est"] + 1e-6)

    # -----------------------
    # 4) Alinhar colunas exatamente como no treino
    #    (cria colunas faltantes com 0 e ordena)
    # -----------------------
    for c in FEATURE_COLS:
        if c not in df_encoded.columns:
            df_encoded[c] = 0.0

    df_encoded = df_encoded[FEATURE_COLS]

    # -----------------------
    # 5) Scale + previsão com Keras
    # -----------------------
    X = df_encoded.values.astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)

    proba_default = float(model.predict(Xs, verbose=0).ravel()[0])
    threshold = 0.5
    pred_label = int(proba_default > threshold)

    return {
        "prob_default": proba_default,
        "threshold": threshold,
        "prediction": pred_label,  # 1 = default, 0 = não default (ajusta ao teu target)
    }
