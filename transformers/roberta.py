from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import torch

# Chargement du modèle pré-entraîné RoBERTa pour la sentiment analysis
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Pipeline de prédiction
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Mapping pour le modèle Roberta
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Exemple sur un texte
example = "the best product I ever owned"
result = pd.DataFrame(sentiment_pipeline(example), columns=["label", "score"])
result["label"] = result["label"].map(label_map)
print(result)