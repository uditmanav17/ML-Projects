import pickle

from src.preprocess import DataProcessing


def predict(X):
    # Load the model and scaler from the saved file
    with open("model/model.pickle", "rb") as f:
        print("Model Imported")
        model, label_encoders, scaler = pickle.load(f)
    preprocessor = DataProcessing()
    # print(X.columns)
    preprocessor.cleaning_steps(X)  # Perform Cleaning
    # print(X.columns)
    # dataprocess.extract_label_value(X)
    preprocessor.perform_feature_engineering(X)  # Perform Feature Engineering

    # Label Encoding
    for column, label_encoder in label_encoders.items():
        X[column] = label_encoder.transform(X[column])

    X = scaler.transform(X)  # Standardize
    return model.predict(X)
