# Variables les plus importantes

# Importation des bibliothèques
import pandas as pd
import numpy as np

# Importation des modèles de classification
from scipy.sparse import load_npz
import pickle

pickle_file = 'Machine Learning/best_model.pkl'
with open(pickle_file, 'rb') as f:
    best_model = pickle.load(f)

# Charger le vectorizer pour obtenir les noms de features
with open('Machine Learning/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Récupérer les noms des features et coefficients
feature_names = np.array(tfidf.get_feature_names_out())
coefs = best_model.coef_[0]

# Créer un DataFrame pour visualisation
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs,
    'Abs_Coef': np.abs(coefs)
}).sort_values('Abs_Coef', ascending=False)

# Afficher les 50 mots les plus influents
print(importance_df.head(50))

# Visualisation graphique
import matplotlib.pyplot as plt

# Créer une figure
fig = plt.figure(figsize=(6, 6))

# Ajouter une sous-figure (axes) à cette figure
ax = fig.add_subplot(1, 1, 1)

# Dessiner le barplot sur ces axes
importance_df.head(20).sort_values('Coefficient').plot.barh(
    x='Feature', 
    y='Coefficient',
    ax=ax,
    title='Top 20 des termes les plus influents'
)

# Sauvegarder la figure en image
fig.savefig("Machine Learning/graphs/importance_20.png", bbox_inches='tight', dpi=300)

# Afficher la figure (facultatif)
plt.show()

