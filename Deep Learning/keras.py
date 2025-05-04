# https://inside-machinelearning.com/premier-projet-keras/
from __future__ import print_function
import pandas as pd

df = pd.read_csv('datasets/Reviews_clean_lemmatized_short.csv')

X = df.loc[:, df.columns != 'Score']
Y = df.loc[:, 'Score']
import keras
# Télécharger les dossiers de formation et de test MNIST
from keras.datasets import mnist
# Télécharger le modèle séquentiel 
from keras.models import Sequential
# Télécharger les couches des cellules neuronales 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras
# Télécharger les dossiers de formation et de test MNIST
from keras.datasets import mnist
# Télécharger le modèle séquentiel 
from keras.models import Sequential
# Télécharger les couches des cellules neuronales 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100)

_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

######################################

df_train = df.sample(frac=0.8)
df_test = df.drop(df_train.index)

X_train = df_train.loc[:, df.columns != 'Score']
Y_train = df_train.loc[:, 'Score']
X_test = df_test.loc[:, df.columns != 'Score']
Y_test = df_test.loc[:, 'Score']

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from tensorflow.keras.utils import plot_model

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=50, batch_size=10)

from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'], color='#066b8b')
plt.plot(history.history['val_accuracy'], color='#b39200')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predictions = model.predict(X_test)

predictions[0]

predictions = (model.predict(X_test) > 0.5).astype(int)

for i in range(5):
	print('%s => Prédit : %d,  Attendu : %d' % (X_test.iloc[i].tolist(), predictions[i], Y_test.iloc[i]))
	
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))