from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os

app = Flask(__name__)
CORS(app)

# Load model
model_path = os.path.join("backend", "models", "diabetes_model.joblib")
svc = joblib.load(model_path)
model = svc["model"]
imputer = svc["imputer"]
features = svc["features"]

def map_prob_to_score(p):
    if p < 0.05: return 0
    if p < 0.15: return 1
    if p < 0.35: return 2
    if p < 0.6: return 3
    if p < 0.8: return 4
    return 5

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    x = []
    for f in features:
        if f not in data:
            return jsonify({"error": f"Missing feature {f}"}), 400
        x.append(float(data[f]))

    x = np.array(x).reshape(1, -1)
    x_imp = imputer.transform(x)
    p = model.predict_proba(x_imp)[0, 1]
    score = map_prob_to_score(p)

    label_map = {
        0: "Stable",
        1: "Low - Monitor",
        2: "Monitor - Take simple steps",
        3: "Action Recommended",
        4: "Clinical review suggested",
        5: "Immediate contact recommended",
    }

    return jsonify({
        "probability": float(p),
        "score": int(score),
        "user_label": label_map[score],
        "short_advice": "Stay consistent with healthy habits." if score <= 2 else "Please consult a healthcare professional.",
        "explanation": "Prediction driven by glucose & BMI (simplified)."
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
