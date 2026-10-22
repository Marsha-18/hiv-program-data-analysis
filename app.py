from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Load the trained machine learning models
classification_model = joblib.load(
    MODEL_DIR / "viral_suppression_model.pkl"
)

regression_model = joblib.load(
    MODEL_DIR / "cd4_prediction_model.pkl"
)

# Load the saved encoders
vl_encoders = joblib.load(
    MODEL_DIR / "viral_suppression_encoders.pkl"
)

cd4_encoders = joblib.load(
    MODEL_DIR / "cd4_encoders.pkl"
)

target_encoder = joblib.load(
    MODEL_DIR / "viral_suppression_target_encoder.pkl"
)


def clean_input(value):
    return " ".join(value.split())


def encode_input(encoder, value, field_name):
    value = clean_input(value)

    for category in encoder.classes_:
        if clean_input(str(category)).casefold() == value.casefold():
            return encoder.transform([category])[0]

    raise ValueError(
        f"Invalid {field_name}: '{value}'. "
        f"Accepted values: {list(encoder.classes_)}"
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():
    try:
        # Collect numerical inputs
        age = float(request.form["age"])
        visits = float(request.form["visits"])
        cd4 = float(request.form["cd4"])
        viral_load = float(request.form["viral_load"])

        # Encode categorical inputs
        gender = encode_input(
            vl_encoders["Gender"],
            request.form["gender"],
            "Gender"
        )

        subcounty = encode_input(
            vl_encoders["SubCounty"],
            request.form["subcounty"],
            "SubCounty"
        )

        artstatus = encode_input(
            vl_encoders["ARTStatus"],
            request.form["artstatus"],
            "ART Status"
        )

        regimen = encode_input(
            vl_encoders["RegimenLine"],
            request.form["regimen"],
            "Regimen Line"
        )

        whostage = encode_input(
            vl_encoders["WHOStage"],
            request.form["whostage"],
            "WHO Stage"
        )

        entrypoint = encode_input(
            vl_encoders["EntryPoint"],
            request.form["entrypoint"],
            "Entry Point"
        )

        adherence = encode_input(
            vl_encoders["AdherenceLevel"],
            request.form["adherence"],
            "Adherence"
        )

        insurance = encode_input(
            vl_encoders["InsuranceCover"],
            request.form["insurance"],
            "Insurance"
        )

        # Prepare features for viral suppression prediction
        suppression_features = pd.DataFrame([{
            "Age": age,
            "ClinicVisits_Last12Months": visits,
            "CD4Count": cd4,
            "Gender": gender,
            "SubCounty": subcounty,
            "ARTStatus": artstatus,
            "RegimenLine": regimen,
            "WHOStage": whostage,
            "EntryPoint": entrypoint,
            "AdherenceLevel": adherence,
            "InsuranceCover": insurance
        }])

        # Predict viral suppression category
        suppression_prediction = classification_model.predict(
            suppression_features
        )

        suppression_result = target_encoder.inverse_transform(
            suppression_prediction
        )[0]

        # Prepare features for CD4 prediction
        cd4_features = pd.DataFrame([{
            "Age": age,
            "ClinicVisits_Last12Months": visits,
            "Gender": gender,
            "SubCounty": subcounty,
            "ARTStatus": artstatus,
            "RegimenLine": regimen,
            "EntryPoint": entrypoint,
            "AdherenceLevel": adherence,
            "InsuranceCover": insurance,
            "ViralLoad_copies_mL": viral_load,
            "WHOStage": whostage
        }])

        # Estimate CD4 count
        cd4_prediction = float(
            regression_model.predict(cd4_features)[0]
        )

        # Display results
        return render_template(
            "result.html",
            suppression_result=suppression_result,
            cd4_prediction=round(cd4_prediction, 2)
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return f"Prediction error: {str(e)}", 400


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)