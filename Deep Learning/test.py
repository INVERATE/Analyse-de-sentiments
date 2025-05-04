# Enregistrez le code dans le fichier « keras-test.py » dans le dossier « keras-test »
from __future__ import print_function
# Télécharger Keras 
import keras
# Télécharger les dossiers de formation et de test MNIST

# Télécharger le modèle séquentiel 
#from keras.models import Sequential
from tensorflow.keras import Sequential
# Télécharger les couches des cellules neuronales 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
df = pd.read_csv('datasets/Reviews_clean_lemmatized_short.csv')

X = df.loc[:, df.columns != 'Score']
Y = df.loc[:, 'Score']

df_train = df.sample(frac=0.8)
df_test = df.drop(df_train.index)

X_train = df_train.loc[:, df.columns != 'Score']
Y_train = df_train.loc[:, 'Score']
X_test = df_test.loc[:, df.columns != 'Score']
Y_test = df_test.loc[:, 'Score']
# Nombre de caractéristiques de données différentes : chiffres 0-9
num_classes = 10
# Nombre de laissez-passer pour la formation du réseau de neurones
epochs = 12
# Nombre de données utilisées lors d’un passage
batch_size = 128
# Dimensions des images d’entrée (28 x 28 pixels par image)
img_rows, img_cols = 28, 28


# Convertir les vecteurs de classe en matrices de classe binaires

# Créer un modèle
model = Sequential()
# Ajouter des couches au modèle
model.add(Conv2D(2, kernel_size=(3, 3),   # couches avec fonction d’activation ReLU
                 activation='relu',
                 ))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compiler le modèle
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
# Former le modèle
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))
# Évaluer le modèle
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])