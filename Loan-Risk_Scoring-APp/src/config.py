from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.pkl"
LOW_THR = 0.20
MED_THR = 0.50
DTI_INPUT_IS_RATIO = True  # your UI computes ratio (0–1)

