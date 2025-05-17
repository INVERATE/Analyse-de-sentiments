from scipy.sparse import load_npz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd


X_train = load_npz("X_train_tfidf.npz").toarray()
Y_train = pd.read_csv("y_train.csv")['y_train']
X_test = load_npz("X_test_tfidf.npz").toarray()
Y_test = pd.read_csv("y_test.csv")['y_test']

# Vérification de la dimension d'entrée
input_dim = X_train.shape[1]  # nombre de colonnes = nombre de features TF-IDF


model = Sequential()
model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=150, batch_size=10)


_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy * 100))