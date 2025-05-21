from keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np

#vérifier GPU
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 1) Chargement et tokenization
text = pd.read_csv("datasets/Reviews_clean_lemmatized_medium.csv")["Text_without_stopwords"]
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

# 2) Padding
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

X = padded_sequences
Y = pd.read_csv("datasets/Reviews_clean_lemmatized_medium.csv")["Score"]

# 3) Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4) One-hot encoding
Y_train_cat = to_categorical(Y_train-1, num_classes=5)
Y_test_cat = to_categorical(Y_test-1, num_classes=5)


# 5) Définition du modèle
model = Sequential([
    # Embedding : transforme chaque mot en vecteur dense de 100 dim
    Embedding(input_dim=vocab_size,
              output_dim=100,
              input_length=max_length,
              mask_zero=True),
    
    # Bidirectional LSTM : capte les dépendances dans les deux sens
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),

    # Deuxième LSTM : hiérarchisation des dépendances plus longues
    LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),

    # Dense : traitement final de la représentation
    Dense(64, activation='relu'),
    Dropout(0.5),

    # Sortie softmax à 5 classes
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['f1_score', 'precision', 'recall', 'categorical_accuracy'])

# Early stopping pour arrêter si la val_loss ne diminue plus
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# 6) Entraînement
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weights = dict(enumerate(class_weights))

# Entraînement avec batch size réduit
history = model.fit(
    X_train, Y_train_cat,
    batch_size=256,
    epochs=30,
    validation_data=(X_test, Y_test_cat),
    callbacks=[early_stop],
    class_weight=class_weights
)


# 7) Évaluation avec multi-class accuracy
_, f1_score, precision, recall, accuracy = model.evaluate(X_test, Y_test_cat)
print(f'Multi-class accuracy: {accuracy:.2%}')
print(f'F1 score: {f1_score}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')


# 8) Rapport de classification
from sklearn.metrics import classification_report
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1) + 1 # Convertir les probabilités en classes
y_true = Y_test

print("\nClassification Report :\n")
print(classification_report(y_true, y_pred, digits=2))

# matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5], normalize='true')
print(cm)

sns.heatmap(cm, annot=True, cmap="viridis")
plt.xlabel("Prediction")
plt.ylabel("Vraie valeur")

# sauvegarder le graphique
plt.savefig("Deep Learning/graphs/confusion_matrix.png")
plt.show()

# Sauvegarde du modèle
model.save('Deep Learning/sentiment_analysis_model.keras')

#history.history
# 8) Visualisation
import matplotlib.pyplot as plt

plt.plot(history.history['categorical_accuracy'], color='#066b8b')
plt.plot(history.history['val_categorical_accuracy'], color='#b39200')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()