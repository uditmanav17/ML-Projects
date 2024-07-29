import pickle
from pathlib import Path

from src.preprocess import DataProcessing

CUR_DIR_PATH = Path(__file__).parent.resolve()


def predict(X):
    """
    Loads the model and scaler from the saved file, performs data preprocessing, feature engineering, label encoding, and standardization on the input data X, and returns the model predictions.

    Args:
    - X: Input data for prediction.

    Returns:
    Model predictions.
    """
    # Load the model and scaler from the saved file
    with open(CUR_DIR_PATH.parent / "model/model.pickle", "rb") as f:
        print("Model Imported")
        model, label_encoders, scaler = pickle.load(f)
    preprocessor = DataProcessing()
    # print(X.columns)
    preprocessor.cleaning_steps(X)
    # print(X.columns)
    # dataprocess.extract_label_value(X)
    preprocessor.perform_feature_engineering(X)

    # Label Encoding
    for column, label_encoder in label_encoders.items():
        X[column] = label_encoder.transform(X[column])

    X = scaler.transform(X)
    return model.predict(X)
