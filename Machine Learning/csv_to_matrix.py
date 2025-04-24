# import required module
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# assign documents
df = pd.read_csv("datasets/Reviews_clean_lemmatized_short.csv")
texts = df['Text_without_stopwords']


# create object
print('TF-IDF Vectorizer processing...')
tfidf = TfidfVectorizer(
    max_features=10000,       # Limite le nombre de caractéristiques
    min_df=5,                 # Ignore les termes trop rares
    max_df=0.9,              # Ignore les termes trop fréquents
)

# Ne pas convertir en matrice dense
result = tfidf.fit_transform(texts)

# Pour l'analyse exploratoire, utiliser des échantillons
print("Exemple de caractéristiques :", tfidf.get_feature_names_out()[:10])
print("Dimension de la matrice TF-IDF :", result.shape)

# get idf values
print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
    print(ele1, ':', ele2)

# get indexing
print('\nWord indexes:')
print(tfidf.vocabulary_)

# display tf-idf values
print('\ntf-idf value:')
print(result)

# in matrix form
# print('\ntf-idf values in matrix form:')
# print(result.toarray())


from scipy.sparse import save_npz

print('Saving sparse matrix...')
save_npz('Machine Learning/tfidf_matrix_sparse.npz', result)