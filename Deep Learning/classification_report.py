import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

#importer model de deep learning
model = keras.models.load_model("Deep Learning/sentiment_analysis_model.keras")

# charger le jeu de données
print("Loading data...")
df = pd.read_csv("datasets/Reviews_test.csv")
X_test = df["Text"]
y_test = df["Score"]

# prediction
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_test)
sequences = tokenizer.texts_to_sequences(X_test)

# 2) Padding
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
y_pred_prob = model.predict(padded_sequences)
y_pred = np.argmax(y_pred_prob, axis=1) + 1


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
plt.savefig("Deep Learning/graphs/confusion_matrix.png")
plt.show()