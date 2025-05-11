import shap
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

# Charger le meilleur modèle
with open('Machine Learning/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Récupérer les noms des features
feature_names = np.array(tfidf.get_feature_names_out())

# Convertir les matrices sparse en dense pour SHAP (attention à la mémoire)
X_train_dense = X_train[:1000].toarray()  # Limiter la taille pour éviter une surcharge mémoire
X_test_dense = X_test.toarray()


# Initialiser l'explainer pour modèle linéaire
explainer = shap.LinearExplainer(
    best_model, 
    X_train, 
    feature_names=feature_names
)

# Calcul des valeurs SHAP pour un échantillon
sample_idx = np.random.choice(X_test.shape[0], 100, replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])

# Visualisation globale
shap.summary_plot(shap_values, X_test[sample_idx], feature_names=feature_names)

# Visualisation individuelle
single_idx = np.random.choice(X_test.shape[0], 1, replace=False)
shap.force_plot(
    explainer.expected_value, 
    shap_values[single_idx], 
    X_test[sample_idx][single_idx], 
    feature_names=feature_names
)
