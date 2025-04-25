# Etape 4 : Modélisation avec du Machine Learning


# Importation des bibliothèques
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
import numpy as np

# Importation des modèles de classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Importation des métriques d'évaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve


# Récupération des données nettoyées
print("Chargement des données...")
data = pd.read_csv('datasets/Reviews_clean_lemmatized_short.csv')

# matrice df-idf
print("Chargement de la matrice...")
sparse_matrix = load_npz('Machine Learning/tfidf_matrix_sparse.npz')
# print("Matrix shape:", sparse_matrix.shape)
# print("Data shape:", data.shape)

# Séparation des données en variables explicatives (X) et variable cible (y)
X = sparse_matrix
y = data['Score']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation des modèles
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier()
}

# définition de l'espace des hyperparamètres possibles
parameters = {
    'Logistic Regression': {'C': np.linspace(0.01, 100, 10)},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': range(1, 10)},
    'K-Nearest Neighbors': {'n_neighbors': range(1, 20)},
    'Random Forest': {'n_estimators': range(10, 100, 10), 'max_depth': range(1, 10)}
    }

# Dictionnaire pour stocker les résultats
results = {
    'Model': [],
    'Accuracy': [],
    'ROC AUC': []
}

# on utilise un gridsearch validation croisée afin de trouver les meilleurs hyperparamètres du modèle sur plusieurs itérations
from sklearn.model_selection import GridSearchCV
best_models = {}

# Boucle sur les modèles
for model_name, model in models.items():
    print(f"\nEntrainement du modèle : {model_name}")
    # Entraînement du modèle
    grid_search = GridSearchCV(model, parameters[model_name], cv=5)
    grid_search.fit(X_train, y_train)
    model.set_params(**grid_search.best_params_)
    print(f"Meilleurs hyperparamètres pour {model_name} : {grid_search.best_params_}")
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    
    # Ajout des résultats au dictionnaire
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)
    results['ROC AUC'].append(roc_auc)
    
    # Sauvegarde du meilleur modèle
    best_models[model_name] = model

# Création d'un DataFrame pour les résultats
results_df = pd.DataFrame(results)

# Affichage des résultats
print(f"\nRésultats des modèles : \n{results_df}\n")


# Utilisation du meilleur modèle
# on choisit le meilleur modèle qui a le meilleur accuracy
best_model = best_models[max(best_models, key=lambda k: best_models[k].score(X_test, y_test))]
print(f"\nUtilisation du meilleur modèle : {best_model}")

# test sur une phrase
sentence = "i read the review i was hestitant as it was reported the drawer did not out easy well it comes out easy i room for plenty of choices in the drawer as family likes hot chocolate tea personally i like coffee the best thing is that it is a space saver i put the kcup machine on it i plemnty of room for condiments with other storage syatems you it next to the kcup machine this takes a lot of spce space is important in cottage which is now home i think this was a great buy for me it is funny how sometimes you buy something it turns out to be a dud this is definitely the money"
sentence_score = 5

# convertir la phrase en vecteur TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

with open('Machine Learning/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
# on s'assure que la phrase soit convertie en vecteur avec les mêmes dimensions que la matrice
sentence_vector = tfidf.transform([sentence]) # PAS FIT !!!

sentence_score_predict = best_model.predict(sentence_vector)

print(f"Score réel : {sentence_score}")
print(f"Score prédit : {sentence_score_predict[0]}")
