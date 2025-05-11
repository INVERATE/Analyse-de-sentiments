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

# Afficher les 20 mots les plus influents
print(importance_df.head(50))

# Visualisation graphique
import matplotlib.pyplot as plt

importance_df.head(50).sort_values('Coefficient').plot.barh(
    x='Feature', 
    y='Coefficient',
    title='Top 20 des termes les plus influents'
)
plt.show()