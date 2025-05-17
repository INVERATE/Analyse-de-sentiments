import pandas as pd
from sklearn.metrics import classification_report
df = pd.read_csv("datasets/Reviews_clean_lemmatized_short_with_predictions.csv")

# classification report
print(classification_report(df["Score"], df["predictions"]))

# matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(df["Score"], df["predictions"])
print(cm)

sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

