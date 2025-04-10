#Nettoyage des colonnes avec LabelEncoder pour modifier les colonnes non numériques
# Suppression des colonnes pas utiles

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Chargement des données
data = pd.read_csv('datasets/Reviews_clean.csv')

# Suppression des colonnes non utiles
data.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator','HelpfulnessDenominator','Time',Summary], inplace=True)