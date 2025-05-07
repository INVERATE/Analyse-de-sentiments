# importer module pandas
import pandas as pd

# lire les données
df = pd.read_csv("datasets/Reviews.csv")

# afficher description
print(df.describe())
print(df.dtypes)

#afficher valeurs vides
print(df.isnull().sum())

#supprimer lignes vides
# df.dropna(inplace=True)