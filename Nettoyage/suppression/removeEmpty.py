import pandas as pd

df = pd.read_csv("datasets/Reviews.csv")

print(df.describe())
print(df.dtypes)

#afficher valeurs vides
print(df.isnull().sum())

#supprimer lignes vides
# df.dropna(inplace=True)