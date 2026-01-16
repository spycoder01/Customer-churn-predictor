print("APP FILE IS EXECUTING")

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("Customer_churn_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


# Exact feature order used during training
FEATURE_ORDER = [
    "gender",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]

# Columns that were label-encoded during training
CATEGORICAL_COLS = ["gender"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        # 1. Read raw inputs (same datatype as training)
        input_data = {
            "gender": request.form["gender"],                 # string
            "tenure": int(request.form["tenure"]),
            "MonthlyCharges": float(request.form["monthly_charges"]),
            "TotalCharges": float(request.form["total_charges"]),
            "SeniorCitizen": int(request.form["senior_citizen"])
        }

        # 2. Convert to DataFrame with correct order
        input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)

        # 3. Apply encoders ONLY to categorical columns
        for col in CATEGORICAL_COLS:
            input_df[col] = encoders[col].transform(input_df[col])

        # 4. Predict
        result = model.predict(input_df)[0]

        prediction = (
            "Customer will CHURN"
            if result == 1
            else "Customer will NOT churn"
        )

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
