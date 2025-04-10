# Etape 4 : Modélisation avec du Machine Learning


# Importation des bibliothèques
import pandas as pd
from sklearn.model_selection import train_test_split

# Importation des modèles de classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Importation des métriques d'évaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# Récupération des données nettoyées
data = pd.read_csv('datasets/Reviews_clean.csv')

# Séparation des données en variables explicatives (X) et variable cible (y)
X = data.drop(columns=['Score'])
y = data['Score']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation des modèles
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Dictionnaire pour stocker les résultats
results = {
    'Model': [],
    'Accuracy': [],
    'ROC AUC': []
}

# Boucle sur les modèles
for model_name, model in models.items():
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Ajout des résultats au dictionnaire
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)
    results['ROC AUC'].append(roc_auc)

# Création d'un DataFrame pour les résultats
results_df = pd.DataFrame(results)

# Affichage des résultats
print(results_df)