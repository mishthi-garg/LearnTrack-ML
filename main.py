from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    return {"status": "alive"}
    
model = joblib.load("grade_predictor_model.pkl")

class PredictionInput(BaseModel):
    overall_z: float

@app.post("/predict")
def predict_grade(input: PredictionInput):
    features = np.array([[input.overall_z]])
    predicted_grade = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    grade_classes = model.classes_
    confidence_scores = {grade: round(float(prob),3) for grade,prob in zip(grade_classes, probabilities)}

    return{
        "predicted_grade": predicted_grade,
        "confidence": confidence_scores
    }
@app.get("/")
def health_check():
    return{
        "status": "LearnTrack ML service running"
    }