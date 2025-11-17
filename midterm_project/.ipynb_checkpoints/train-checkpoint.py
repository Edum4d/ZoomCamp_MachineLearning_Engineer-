import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import json
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load full training data (train + validation merged beforehand)
df_full_train = pd.read_csv("train.csv")

# ---------------------------------
# 1) Separate target
# ---------------------------------
y_full_train = df_full_train.loan_paid_back
del df_full_train["loan_paid_back"]

# ---------------------------------
# 2) Repeat feature encoding / engineering on df_full_train
#    (same logic as you used before, but on the full train)
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

# Derived features
df_full_train["loan_to_income"]     = df_full_train["loan_amount"] / df_full_train["annual_income"]
df_full_train["total_debt_est"]     = (df_full_train["annual_income"] * df_full_train["debt_to_income_ratio"]) / 100
df_full_train["loan_to_total_debt"] = df_full_train["loan_amount"] / (df_full_train["total_debt_est"] + 1e-6)

# ---------------------------------
# 3) DictVectorizer on the entire train
# ---------------------------------
dv = DictVectorizer(sparse=False)

train_full_dict = df_full_train.to_dict(orient="records")
X_train_full = dv.fit_transform(train_full_dict)

# ---------------------------------
# 4) Load BEST hyperparameters from JSON
# ---------------------------------
with open("xgb_best_params.json", "r") as f:
    best_cfg = json.load(f)

# Extract num_boost_round (if present) or use default
num_boost_round = best_cfg.pop("num_boost_round", 200)

# Remove metrics that should not be passed to XGBoost
for m in ["val_auc", "f1"]:
    best_cfg.pop(m, None)

best_params = best_cfg  # now contains only valid XGBoost params

# ---------------------------------
# 4b) Train final XGBoost model with BEST hyperparameters
# ---------------------------------
features = list(dv.get_feature_names_out())
dtrain_full = xgb.DMatrix(X_train_full, label=y_full_train, feature_names=features)

model_final = xgb.train(
    params=best_params,
    dtrain=dtrain_full,
    num_boost_round=num_boost_round
)

# ---------------------------------
# 5) Save model + DictVectorizer + encoders (for production use)
# ---------------------------------
model_final.save_model("xgb_model_final.json")

with open("dv_xgb.pkl", "wb") as f:
    pickle.dump(dv, f)

with open("one_hot_encoder.pkl", "wb") as f:
    pickle.dump(one_hot, f)

with open("ordinal_encoder.pkl", "wb") as f:
    pickle.dump(ordinal, f)

print("Final model trained on FULL train and saved to 'xgb_model_final.json'")
print("DictVectorizer and encoders saved to 'dv_xgb.pkl', 'one_hot_encoder.pkl', 'ordinal_encoder.pkl'")
print("Hyperparameters loaded from 'xgb_best_params.json':", best_params)
print("num_boost_round:", num_boost_round)
