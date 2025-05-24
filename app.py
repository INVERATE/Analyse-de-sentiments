from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger modèle + tokenizer
model_path = "inverate/roberta-fine-tuned-web"
import os
from dotenv import load_dotenv

load_dotenv()
token_huggingface = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=token_huggingface)
model = AutoModelForSequenceClassification.from_pretrained(model_path, use_auth_token=token_huggingface)
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# App FastAPI
# App FastAPI avec CORS
app = FastAPI(title="API Sentiment Analysis", version="1.0")

# Configuration CORS essentielle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (à ajuster en production)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes
    allow_headers=["*"],  # Autorise tous les en-têtes
)

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
