import pandas as pd
from scipy.sparse import load_npz
import numpy as np
import pickle

# Chargement des données
X_train = load_npz('Machine Learning/X_train_tfidf.npz')
X_test = load_npz('Machine Learning/X_test_tfidf.npz')
y_train = pd.read_csv('Machine Learning/y_train.csv')['y_train']
y_test = pd.read_csv('Machine Learning/y_test.csv')['y_test']

# Charger le vectorizer pour obtenir les noms de features
with open('Machine Learning/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Récupérer les noms des features et coefficients
feature_names = np.array(tfidf.get_feature_names_out())

# Ré-entraîner avec régularisation L1
from sklearn.linear_model import LogisticRegression

# Modèle de régularisation L1 avec les memes hyperparamètres
l1_model = LogisticRegression(
    penalty='l1', 
    solver='liblinear', 
    C=11.12,  # Contrôle la force de régularisation
    random_state=42
).fit(X_train, y_train)

# Features sélectionnées (coefficients non-nuls) triées par ordre décroissant de poids des coefficients
selected_features = feature_names[l1_model.coef_[0] != 0]
selected_features = selected_features[np.argsort(-abs(l1_model.coef_[0][l1_model.coef_[0] != 0]))]
print(selected_features)