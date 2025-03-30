import pandas as pd

# Ce script va retirer les mots dont la distribution est similaire de la distribution des notes
# cela permet de ne garder que des mots qui sont significatifs pour la prediction de la note

#%% fonctions
def NormalizeDistribution(distribution):
    for score in range(1, 6):
        if score not in distribution:
            distribution[score] = 0
    
    total = sum(distribution.values())
    return {k: v / total for k, v in distribution.items()}


def CompareDistribution(dict_word, dict_score):
    # renvoie un pourcentage de différence entre deux listes normalisées
    # on va comparer la distribution des mots et la distribution des notes
    total_offset = 0
    for score in range(1, 6):
        str_score = str(score)
        total_offset += abs(dict_word[str_score] - dict_score[str_score])
    return total_offset


#%% paramètres
df = pd.read_csv("Nettoyage/wordDistribution/most_common_words.csv")
offset_threshold = 0.05  # seuil de tolérance pour la distribution des mots
score_distribution = {"1": 52268, "2": 29769, "3": 42640, "4": 80655, "5": 363122}
score_distribution_normalized = NormalizeDistribution(score_distribution)

#%% script
list_stopwords = []

for index, row in df.iterrows():
    # si la distribution des mots est similaire à la distribution des notes, on le retire

    word_distribution_normalized = NormalizeDistribution(row[["1", "2", "3", "4", "5"]].to_dict())
    offset = CompareDistribution(word_distribution_normalized, score_distribution_normalized)
    
    #print(offset, row["word"])
    if offset < offset_threshold:
        print(offset, row["word"])
        list_stopwords.append(row["word"])

print(list_stopwords)

#%% sauvegarde
print("Saving stopwords...")
with open("Nettoyage/wordDistribution/stopwords.txt", "w") as f:
    for word in list_stopwords:
        f.write(word + "\n")