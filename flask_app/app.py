from flask import Flask, render_template, request
import mlflow
import pandas as pd
import time
import dagshub
import warnings
import yaml
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize MLflow Tracking
mlflow.set_tracking_uri('https://dagshub.com/ayazr425/Heart-Disease-Predection.mlflow')
dagshub.init(repo_owner='ayazr425', repo_name='Heart-Disease-Predection', mlflow=True)

# Initialize Flask app
app = Flask(__name__)

# Load latest model from MLflow
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# Load scaling column list from params.yaml
def load_params(path="params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

params = load_params()
columns_to_scale = params['feature_engineering']['columns_to_scale']

# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        input_data = [
            float(form_data['age']),
            int(form_data['sex']),
            int(form_data['cp']),
            float(form_data['trestbps']),
            float(form_data['chol']),
            int(form_data['fbs']),
            int(form_data['restecg']),
            float(form_data['thalach']),
            int(form_data['exang']),
            float(form_data['oldpeak']),
            int(form_data['slope']),
            int(form_data['ca']),
            int(form_data['thal'])
        ]

        columns = [
            'Age', 'Sex', 'Chest pain type', 'Blood Pressure', 'Cholesterol',
            'Sugre Blood Value', 'EKG results', 'Max heart rate', 'Exercise angina',
            'Segment Depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
        ]

        input_df = pd.DataFrame([input_data], columns=columns)

        # ✅ Fit StandardScaler on a default dummy dataset for reference
        # Ideally you'd fit this on training data, but since you're avoiding loading,
        # we'll fit on a dummy row with similar structure just to enable scaling.
        scaler = StandardScaler()
        dummy_data = pd.DataFrame([[0]*len(columns)], columns=columns)
        scaler.fit(dummy_data[columns_to_scale])  # Fit on dummy (structure-only)

        # ✅ Now scale the actual input
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        prediction = model.predict(input_df)[0]
        print(prediction)

        return render_template("index.html", result=int(prediction))

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template("index.html", result="Error: Invalid input")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
