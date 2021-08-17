import tensorflow as tf
import pandas as pd
import joblib

# load saved standardising object
scaler = joblib.load("data/scaler.joblib")

# load saved keras model
model = tf.keras.models.load_model("data/banknote_authentication_model.h5")


def load_model(path: str):
    global scaler, model

    scaler = joblib.load(f"{path}/scaler.joblib")
    model = tf.keras.models.load_model(f"{path}/banknote_authentication_model.h5")


def get_prediction(data: dict):
    """
    A function that reshapes the incoming JSON data, loads the saved model objects
    and returns the predicted class and probability.

    :param data: Dict with keys representing features and values representing the associated value
    :return: Dict with keys 'predicted_class' (class predicted) and 'predicted_prob' (probability of prediction)
    """

    # convert new data dict as a DataFrame and reshape the columns to suit the model
    new_data = {k: [v] for k, v in data.items()}
    new_data_df = pd.DataFrame.from_dict(new_data)
    new_data_df = new_data_df[
        [
            "variance_of_wavelet",
            "skewness_of_wavelet",
            "curtosis_of_wavelet",
            "entropy_of_wavelet",
        ]
    ]

    # scale new data using the loaded object
    x = scaler.transform(new_data_df.values)

    # make new predictions frm the newly scaled data
    preds = model.predict(x)

    return preds
