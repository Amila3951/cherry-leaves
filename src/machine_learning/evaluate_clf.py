import joblib
import json

def load_test_evaluation(version):
    try:
        with open(f"outputs/{version}/evaluate.json", "r") as f:
            evaluation = json.load(f)
        evaluation['confusion_matrix'] = joblib.load(f"outputs/{version}/confusion_matrix.joblib")
        return evaluation
    except FileNotFoundError:
        return None