from src.features import build_row
from src.predict import load_model, score
def test_predict_runs():
    _ = load_model()
    r = build_row(30, 60000, 200000, 720, 24, 6.5, 360, 0.35, 0)
    prob, label = score(r)
    assert 0.0 <= prob <= 1.0
    assert label in {"Low Risk ✅","Medium Risk ⚠️","High Risk ❌"}
