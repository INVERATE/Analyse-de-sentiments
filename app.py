from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger modèle + tokenizer
model_path = "./fine_tuned_roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# App FastAPI
app = FastAPI(title="API Sentiment Analysis", version="1.0")

# Schéma de requête
class ReviewRequest(BaseModel):
    text: str

# Route d'accueil
@app.get("/")
def root():
    return {"message": "API Roberta Sentiment Analysis active"}

# Route de prédiction
@app.post("/predict")
def predict_sentiment(data: ReviewRequest):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    return {
        "predicted_score": pred + 1,  # Labels vont de 0 à 4
        "confidence": round(confidence, 4)
    }
