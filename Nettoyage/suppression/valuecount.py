import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Chargement des données
df = pd.read_csv("datasets/Reviews.csv")

# Comptage des scores
score_counts = df["Score"].value_counts().sort_index()

# Couleurs viridis
colors = cm.viridis([i / len(score_counts) for i in range(len(score_counts))])

# Création du camembert
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    score_counts,
    labels=score_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={'edgecolor': 'white'}
)

# Personnalisation du texte
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_color('white')

# Titre
plt.title("Répartition des scores dans le dataset", fontsize=16)
plt.axis('equal')  # Rend le cercle proportionnel
plt.tight_layout()
plt.show()
