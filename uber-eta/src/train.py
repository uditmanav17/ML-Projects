# %%
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from preprocess import DataProcessing

# %% load and prepare data
df_train = pd.read_csv("../data/train.csv")
df_train.head()
dp = DataProcessing()
dp.cleaning_steps(df_train)
dp.extract_label_value(df_train)
dp.perform_feature_engineering(df_train)

# %% Split features & label
X = df_train.drop("time_taken_min", axis=1)  # Features
y = df_train["time_taken_min"]  # Target variable

# %%
label_encoders = dp.label_encoding(X)  # Label Encoding
X_train, X_test, y_train, y_test = dp.data_split(X, y)  # Test Train Split
X_train, X_test, scaler = dp.standardize(X_train, X_test)  # Standardization
# %% Build Model
model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
model.fit(X_train, y_train)

# %% Evaluate Model
y_pred = model.predict(X_test)
dp.evaluate_model(y_test, y_pred)

# %% Create model.pkl and Save Model
model_save_path = Path("../model")
model_save_path.mkdir(exist_ok=True, parents=True)
with open(model_save_path / "model.pickle", "wb") as f:
    pickle.dump((model, label_encoders, scaler), f)
print("Model pickle saved to model folder")
