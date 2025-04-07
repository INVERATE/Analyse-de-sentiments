#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#%%

df = pd.read_csv(r'C:\Users\fanny\OneDrive\Bureau\Ingé2\MACHINE learning avancé\TP3\::::.csv')

# Boxplot pour voir si normalisation utile
df.boxplot()
plt.show()

X = df.drop(['score'], axis=1).values
y = df['score'].values


normelized_X = preprocessing.normalize(X)


X_train, X_test, y_train, y_test = train_test_split(normelized_X, y, test_size=0.2, random_state=42)
