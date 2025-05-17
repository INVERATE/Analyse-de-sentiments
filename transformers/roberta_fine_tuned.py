from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import torch

def predict_sentiment(text):
    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Envoi sur GPU si dispo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Probabilités + prédiction
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=1).item()

    return {
        "predicted_score": pred_class + 1,  # car labels vont de 0 à 4
        "probabilities": probs.squeeze().cpu().numpy()
    }


# Chargement du modèle pré-entraîné RoBERTa pour la sentiment analysis
model_path = "./fine_tuned_roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Exemple
exemple = "Love it!,i very satisfied with twizzler purchase i share with others we all enjoy i will definitely be order more"
result = predict_sentiment(exemple)
print("Prédiction :", result)

# prédiction sur un dataframe
df = pd.read_csv("datasets/Reviews_clean_lemmatized_short.csv")
df["predictions"] = df["Text_without_stopwords"].apply(lambda x: predict_sentiment(x)["predicted_score"])


# Sauvegarde des prédictions
df.to_csv("datasets/Reviews_clean_lemmatized_short_with_predictions.csv", index=False)

from sklearn.metrics import classification_report

# Assure-toi que les colonnes existent :
print(classification_report(df["Score"], df["predictions"]))