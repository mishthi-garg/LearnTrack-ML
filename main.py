from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    return {"status": "alive"}
    
saved = joblib.load("grade_predictor_model_v2.pkl")
model = saved["model"]
GRADE_ORDER = saved["grade_order"]  # e.g. ["F","D","CD","C","BC","B","AB","A"]

class PredictionInput(BaseModel):
    overall_z: float

@app.post("/predict")
def predict_grade(input: PredictionInput):
    features = np.array([[input.overall_z]])

    # Get every individual tree's prediction instead of just the forest average.
    # This replaces predict_proba, which regressors don't have.
    tree_preds = np.array([tree.predict(features)[0] for tree in model.estimators_])

    raw_pred = tree_preds.mean()
    rounded_idx = int(np.clip(round(raw_pred), 0, len(GRADE_ORDER) - 1))
    predicted_grade = GRADE_ORDER[rounded_idx]

    # Confidence: what fraction of trees "voted" for the same rounded grade
    tree_rounded = np.clip(np.round(tree_preds), 0, len(GRADE_ORDER) - 1).astype(int)
    vote_counts = np.bincount(tree_rounded, minlength=len(GRADE_ORDER))
    confidence_scores = {
        GRADE_ORDER[i]: round(float(vote_counts[i] / len(model.estimators_)), 3)
        for i in range(len(GRADE_ORDER))
        if vote_counts[i] > 0
    }
    
    return{
        "predicted_grade": predicted_grade,
        "confidence": confidence_scores
    }
@app.get("/")
def health_check():
    return{
        "status": "LearnTrack ML service running"
    }