import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import pickle

# Charger les données
df = pd.read_csv("datasets/Reviews_clean_lemmatized_short.csv")
texts = df['Text_without_stopwords']
y = df['Score']

# Séparer avant TF-IDF
X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# Créer le vectorizer
print("TF-IDF Vectorizer training uniquement sur le jeu d'entraînement...")
tfidf = TfidfVectorizer(max_features=10000)

# Fit sur le train, transform sur train et test
X_train_tfidf = tfidf.fit_transform(X_train_texts)
X_test_tfidf = tfidf.transform(X_test_texts)

# Sauvegarde
save_npz('Machine Learning/X_train_tfidf.npz', X_train_tfidf)
save_npz('Machine Learning/X_test_tfidf.npz', X_test_tfidf)
pd.DataFrame({'y_train': y_train}).to_csv('Machine Learning/y_train.csv', index=False)
pd.DataFrame({'y_test': y_test}).to_csv('Machine Learning/y_test.csv', index=False)

with open('Machine Learning/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
