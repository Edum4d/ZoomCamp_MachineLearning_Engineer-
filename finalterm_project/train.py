import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import pickle
import json
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------------------------
# GPU setup
# ---------------------------
print("TF version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision:", mixed_precision.global_policy())
    except Exception as e:
        print("Mixed precision not enabled:", e)

DEVICE = "/GPU:0" if gpus else "/CPU:0"
print("Using device:", DEVICE)

# ---------------------------------
# 1) Load full training data (train + validation merged beforehand)
# ---------------------------------
df_full_train = pd.read_csv("train.csv")

# ---------------------------------
# 2) Separate target
# ---------------------------------
y_full_train = df_full_train.loan_paid_back.values.astype(np.int32)
df_full_train = df_full_train.drop(columns=["loan_paid_back"])

# ---------------------------------
# 3) Repeat feature encoding / engineering on df_full_train
# ---------------------------------
ordinal_cols = ["grade_subgrade", "loan_purpose", "employment_status", "education_level"]
one_hot_cols = ["gender", "marital_status"]

one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

one_hot_arr = one_hot.fit_transform(df_full_train[one_hot_cols])
ordinal_arr = ordinal.fit_transform(df_full_train[ordinal_cols])

one_hot_df = pd.DataFrame(
    one_hot_arr,
    columns=one_hot.get_feature_names_out(one_hot_cols),
    index=df_full_train.index
)

ordinal_df = pd.DataFrame(
    ordinal_arr,
    columns=ordinal_cols,
    index=df_full_train.index
)

df_full_train = pd.concat(
    [
        df_full_train.drop(one_hot_cols + ordinal_cols, axis=1),
        ordinal_df,
        one_hot_df
    ],
    axis=1
)

# Derived features (igual ao teu)
df_full_train["loan_to_income"]     = df_full_train["loan_amount"] / df_full_train["annual_income"]
df_full_train["total_debt_est"]     = (df_full_train["annual_income"] * df_full_train["debt_to_income_ratio"]) / 100
df_full_train["loan_to_total_debt"] = df_full_train["loan_amount"] / (df_full_train["total_debt_est"] + 1e-6)

# ---------------------------------
# 4) Build X (dense numpy) + scale (NN precisa disto)
# ---------------------------------
X_train_full = df_full_train.values.astype(np.float32)

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full).astype(np.float32)

# ---------------------------------
# 5) Load BEST hyperparameters from JSON (nn_best_params.json)
# ---------------------------------
with open("nn_best_params.json", "r") as f:
    best_cfg = json.load(f)

# Remover métricas/extra keys (se existirem)
for k in ["accuracy", "precision", "recall", "f1", "val_auc", "device"]:
    best_cfg.pop(k, None)

# Defaults caso faltem
units = tuple(best_cfg.get("units", (128, 64, 32, 16)))
dropout = float(best_cfg.get("dropout", 0.3))
l2 = float(best_cfg.get("l2", 1e-4))
lr = float(best_cfg.get("lr", 1e-3))
batch_size = int(best_cfg.get("batch_size", 256))

# Podes guardar epochs “ótimos” no JSON; senão usamos um default com early stopping
max_epochs = int(best_cfg.get("max_epochs", 20))
patience = int(best_cfg.get("patience", 10))

print("Loaded best NN params:", {"units": units, "dropout": dropout, "l2": l2, "lr": lr, "batch_size": batch_size})

# ---------------------------------
# 6) Build + Train final Neural Net on FULL train
# ---------------------------------
def build_model(input_dim, units, dropout, l2, lr):
    inp = keras.Input(shape=(input_dim,))
    x = inp

    for u in units:
        x = layers.Dense(u, use_bias=False, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.1)(x)
        x = layers.Dropout(dropout)(x)

    # IMPORTANTE com mixed precision: output float32
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        jit_compile=True  # se der erro na tua versão, remove
    )
    return model

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True, verbose=1),
]

with tf.device(DEVICE):
    model_final = build_model(
        input_dim=X_train_full_scaled.shape[1],
        units=units,
        dropout=dropout,
        l2=l2,
        lr=lr
    )

    model_final.fit(
        X_train_full_scaled, y_full_train,
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,         # aqui podes deixar 0 se quiseres silencioso
        callbacks=callbacks
    )

# ---------------------------------
# 7) Save model + preprocessors (for production use)
# ---------------------------------
model_final.save("nn_model_final.keras")

with open("scaler_nn.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("one_hot_encoder.pkl", "wb") as f:
    pickle.dump(one_hot, f)

with open("ordinal_encoder.pkl", "wb") as f:
    pickle.dump(ordinal, f)

# também pode ser útil salvar a lista de colunas finais (ordem importa!)
with open("nn_feature_columns.json", "w") as f:
    json.dump(list(df_full_train.columns), f, indent=2)

print("Final NN trained on FULL train and saved to 'nn_model_final.keras'")
print("Scaler saved to 'scaler_nn.pkl'")
print("Encoders saved to 'one_hot_encoder.pkl', 'ordinal_encoder.pkl'")
print("Feature columns saved to 'nn_feature_columns.json'")
print("Hyperparameters loaded from 'nn_best_params.json'")
