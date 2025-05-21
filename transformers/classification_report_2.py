import pandas as pd
from sklearn.metrics import classification_report
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


#charger le modèle roberta local
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
print("Loading model...")
model_path = "./fine_tuned_roberta_local"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# charger le jeu de données
print("Loading data...")
df = pd.read_csv("datasets/Reviews_test.csv")
X_test = df["Text"]
y_test = df["Score"]
y_pred = [predict_sentiment(text)["predicted_score"] for text in X_test]


# classification report
print("\nClassification Report :\n")
print(classification_report(y_test, y_pred))

# matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5], normalize='true')
print(cm)

sns.heatmap(cm, annot=True, cmap="viridis")
plt.xlabel("Prediction")
plt.ylabel("Vraie valeur")

# sauvegarder le graphique
plt.savefig("transformers/graphs/confusion_matrix.png")
plt.show()