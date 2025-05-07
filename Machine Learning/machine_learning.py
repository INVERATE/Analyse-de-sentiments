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


with open('Machine Learning/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

sentence = "i read the review i was hestitant..."
sentence_vector = tfidf.transform([sentence])
predicted_score = best_model.predict(sentence_vector)

print(f"Score prédit : {predicted_score[0]}")