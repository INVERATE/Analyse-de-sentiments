# Etape 4 : Modélisation avec du Machine Learning
#%%

# Importation des bibliothèques
import pandas as pd
from scipy import sparse
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
import matplotlib.pyplot as plt
import seaborn as sns


# importation pour shap
import shap
import matplotlib.pyplot as plt

# Récupération des données nettoyées
print("Loading data...")
data = pd.read_csv('datasets/Reviews_clean_lemmatized_short.csv')

# matrice df-idf
print("Loading sparse matrix...")
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
    print(f"Training {model_name}...")
    # Entraînement du modèle
    grid_search = GridSearchCV(model, parameters[model_name], cv=5)
    grid_search.fit(X_train, y_train)
    model.set_params(**grid_search.best_params_)
    print("best param : ",grid_search.best_params_)
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
print(results_df)

"""
# Utilisation du meilleur modèle
model = best_models['Logistic Regression']

# test sur une phrase
sentence = "i read the review i was hestitant as it was reported the drawer did not out easy well it comes out easy i room for plenty of choices in the drawer as family likes hot chocolate tea personally i like coffee the best thing is that it is a space saver i put the kcup machine on it i plemnty of room for condiments with other storage syatems you it next to the kcup machine this takes a lot of spce space is important in cottage which is now home i think this was a great buy for me it is funny how sometimes you buy something it turns out to be a dud this is definitely the money"
sentence_score = 5
# convertir la phrase en vecteur TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000)
sentence_vector = tfidf.fit_transform([sentence])

sentence_score_predict = model.predict(sentence_vector)

print(sentence_score_predict)
"""

#%%
"""
print("Loading data...")
data = pd.read_csv('./../../datasets/Reviews_clean_lemmatized_short.csv')

# matrice df-idf
print("Loading sparse matrix...")
sparse_matrix = load_npz('./../../Machine Learning/tfidf_matrix_sparse.npz')
# print("Matrix shape:", sparse_matrix.shape)
# print("Data shape:", data.shape)
df = pd.read_csv('./../../datasets/Reviews_clean_lemmatized_short.csv')
# Séparation des données en variables explicatives (X) et variable cible (y)
X = sparse_matrix
y = data['Score']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etude de l'importance des variables pour random forest avec shap
# utilisation des meilleurs hyperparametre
rf_clf = RandomForestClassifier(max_depth=8, n_estimators =10)

rf_clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = rf_clf.predict(X_test)


# load JS visualization code to notebook
shap.initjs()

# Create the explainer

explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_train)

explainer = shap.TreeExplainer(model, data=sparse_matrix[:100].todense())
shap_values = explainer.shap_values(sparse_matrix[:100].todense())
print(f"Variable Importance Plot - Global Interpretation for {model_name}")
figure = plt.figure()
shap.summary_plot(shap_values, X_test)


# Import the LimeTabularExplainer module
from lime.lime_tabular import LimeTabularExplainer

# Get the class names
class_names = ['1', '5']

# Get the feature names
#feature_names = list(X_train.columns)
feature_names = list(df.columns)
# Fit the Explainer on the training data set using the LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names =     
                                 feature_names,
                                 class_names = class_names, 
                                 mode = 'classification')

def display_feat_imp_rforest(model):
    feature_imp = model_3_xgboost.get_booster().get_score(importance_type='weight')
    keys = list(feature_imp.keys())
    values = list(feature_imp.values())
    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    data.nlargest(40, columns="score").plot(kind='barh', figsize = (10,5)) 
display_feat_imp_rforest(model_3_xgboost)

def display_feat_imp_rforest(rforest):
    feat_imp = rforest.feature_importances_
    df_featimp = pd.DataFrame(feat_imp, columns = {"Feature Importance"})
    df_featimp["Feature Name"] = df.columns
    df_featimp = df_featimp.sort_values(by="Feature Importance", ascending=False)
    print(df_featimp)
    df_featimp.plot.barh(y="Feature Importance", x="Feature Name", title="Feature importance", color="red")
display_feat_imp_rforest(rf_clf)
"""