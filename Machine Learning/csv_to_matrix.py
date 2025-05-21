import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from imblearn.over_sampling import RandomOverSampler
import pickle

# 1. Charger les données
df = pd.read_csv("datasets/Reviews_clean_lemmatized_short.csv")
texts = df['Text_without_stopwords']
y = df['Score']

# 2. Split train/test avec stratification pour garder la proportion dans le test
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF Vectorizer sur le jeu d'entraînement uniquement
print("TF-IDF Vectorizer training uniquement sur le jeu d'entraînement...")
tfidf = TfidfVectorizer(max_features=10000)

X_train_tfidf = tfidf.fit_transform(X_train_texts)
X_test_tfidf = tfidf.transform(X_test_texts)

# 4. Rééquilibrage du jeu d'entraînement par sur-échantillonnage
print("Oversampling du jeu d'entraînement pour équilibrer les classes...")
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_tfidf, y_train)

# 5. Sauvegardes
save_npz('Machine Learning/X_train_tfidf_oversampled.npz', X_train_resampled)
save_npz('Machine Learning/X_test_tfidf.npz', X_test_tfidf)
pd.DataFrame({'y_train': y_train_resampled}).to_csv('Machine Learning/y_train_oversampled.csv', index=False)
pd.DataFrame({'y_test': y_test}).to_csv('Machine Learning/y_test.csv', index=False)

with open('Machine Learning/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
