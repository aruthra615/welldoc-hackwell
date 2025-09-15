import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib, os, sys

CSV = "backend/data/diabetes.csv"
if not os.path.exists(CSV):
    print("Missing backend/data/diabetes.csv — please place your CSV there (or run the synthetic fallback).")
    sys.exit(1)

df = pd.read_csv(CSV)

# expected features - change here if your CSV uses different headers
features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
for f in features:
    if f not in df.columns:
        print(f"Missing column {f} in CSV. Please rename CSV headers or adjust the features list in this script.")
        sys.exit(1)

X = df[features]
y = df["Outcome"].astype(int)

imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
clf = CalibratedClassifierCV(pipe, cv=5, method="isotonic")
clf.fit(X_imp, y)

probs = clf.predict_proba(X_imp)[:, 1]
print("AUC:", round(roc_auc_score(y, probs), 3), "Brier:", round(brier_score_loss(y, probs), 3))

os.makedirs("backend/models", exist_ok=True)
joblib.dump({"model": clf, "imputer": imputer, "features": features}, "backend/models/diabetes_model.joblib")
print("Saved model → backend/models/diabetes_model.joblib")
