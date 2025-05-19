from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd

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
    Embedding(input_dim=vocab_size,
              output_dim=100,
              input_length=max_length,   # <-- ici aligné sur max_length
              mask_zero=True),           # mask_zero suffit, pas besoin de Masking explicite
    LSTM(128, return_sequences=False, dropout=0.1),
    Dense(64, activation='relu'),
    Dropout(0.5), # sert à limiter le surapprentissage
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# 6) Entraînement
history = model.fit(
    X_train, Y_train_cat,
    batch_size=512,    # batch plus raisonnable
    epochs=30,
    validation_data=(X_test, Y_test_cat)
)

# 7) Évaluation avec multi-class accuracy
_, accuracy = model.evaluate(X_test, Y_test_cat)
print(f'Multi-class accuracy: {accuracy:.2%}')


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
