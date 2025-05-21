import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import pickle

# Utilisation du meilleur modèle pour prédire un commentaire
with open('Machine Learning/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('Machine Learning/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)


# charger le jeu de données
print("Loading data...")
df = pd.read_csv("datasets/Reviews_test.csv")
X_test = df["Text"]
y_test = df["Score"]

# prediction
X_test_vectorized = tfidf.transform(X_test)
y_pred = best_model.predict(X_test_vectorized)


# classification report
print("\nClassification Report :\n")
print(classification_report(y_test, y_pred, digits=2))

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
plt.savefig("Machine Learning/graphs/confusion_matrix.png")
plt.show()