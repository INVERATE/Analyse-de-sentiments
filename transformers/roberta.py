import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

# === Chargement du modèle Roberta pré-entraîné (3 classes : neg, neutre, pos)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# === Fonction de prédiction pour un texte
def predict_sentiment_roberta_base(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=1).item()
    return pred_class  # 0 = negative, 1 = neutral, 2 = positive

# === Chargement du dataset
df = pd.read_csv("datasets/Reviews_clean_lemmatized_medium.csv")

# === Mapping de Score (1 à 5) vers 0 = neg / 1 = neutre / 2 = pos pour comparer
def map_score_to_sentiment(score):
    if score <= 2:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df["true_sentiment"] = df["Score"].apply(map_score_to_sentiment)

# === Application du modèle
print("Application du modèle RoBERTa base sur les textes...")
df["predicted_sentiment"] = df["Text_without_stopwords"].apply(predict_sentiment_roberta_base)

# === Évaluation des performances
print("\nRapport de classification (3 classes : Négatif / Neutre / Positif) :")
print(classification_report(df["true_sentiment"], df["predicted_sentiment"], target_names=["Negative", "Neutral", "Positive"]))

# === Matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(df["true_sentiment"], df["predicted_sentiment"], normalize='true')
print(cm)

sns.heatmap(cm, annot=True, cmap="viridis")
plt.xlabel("Prediction")
plt.ylabel("Vraie Valeur")
plt.show()

# === Sauvegarde des résultats
df.to_csv("datasets/Reviews_with_roberta_base_predictions.csv", index=False)
