import joblib
from .config import MODEL_PATH, LOW_THR, MED_THR

_model = None
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def score(df_row):
    prob = float(load_model().predict_proba(df_row)[0, 1])
    if prob < LOW_THR:  label = "Low Risk ✅"
    elif prob < MED_THR: label = "Medium Risk ⚠️"
    else: label = "High Risk ❌"
    return prob, label
