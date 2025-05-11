# Etape 4 : Modélisation avec du Machine Learning
#%%

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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import load_npz
import pandas as pd
import numpy as np
from time import perf_counter
import pickle

# Chargement des données
X_train = load_npz('Machine Learning/X_train_tfidf.npz')
X_test = load_npz('Machine Learning/X_test_tfidf.npz')
y_train = pd.read_csv('Machine Learning/y_train.csv')['y_train']
y_test = pd.read_csv('Machine Learning/y_test.csv')['y_test']

# Modèles
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier()
}

parameters = {
    'Logistic Regression': {'C': np.linspace(0.01, 100, 10)},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': range(1, 10)},
    'K-Nearest Neighbors': {'n_neighbors': range(1, 20)},
    'Random Forest': {'n_estimators': range(10, 100, 10), 'max_depth': range(1, 10)}
}

results = {'Model': [], 'Accuracy': [], 'ROC AUC': [], 'Time': []}
best_models = {}

# Entraînement et évaluation
for name, model in models.items():
    print(f"\n{ name } - Entraînement...")
    grid_search = GridSearchCV(model, parameters[name], cv=5)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    time_start = perf_counter()
    best_model.fit(X_train, y_train)
    time_stop = perf_counter()
    time_model = time_stop - time_start
    print(f"Temps d'entraînement : {round(time_model,2)} secondes")
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    results['Model'].append(name)
    results['Accuracy'].append(acc)
    results['ROC AUC'].append(roc)
    results['Time'].append(time_model)
    best_models[name] = best_model

results_df = pd.DataFrame(results)
print("\nRésultats des modèles :")
print(results_df)

# Meilleur modèle
best_model_name = results_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
best_model = best_models[best_model_name]
print(f"\nMeilleur modèle retenu : {best_model_name}")
print(f"Hyperparamètres : {best_model.get_params()}")

# Sauvegarde du meilleur modèle
print("\nSauvegarde du meilleur modèle...")
with open('Machine Learning/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


# Utilisation du meilleur modèle pour prédire un commentaire
with open('Machine Learning/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

sentence = "i read the review i was hestitant..."
sentence_vector = tfidf.transform([sentence])
predicted_score = best_model.predict(sentence_vector)

print(f"Predicted score : {predicted_score}")


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