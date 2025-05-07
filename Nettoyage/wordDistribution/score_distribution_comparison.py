import matplotlib.pyplot as plt
import pandas as pd

def NormalizeDistribution(distribution):
    total = sum(distribution.values())
    return {k: v / total for k, v in distribution.items()}

def ExtractSPecificWordDistribution(word, data="Nettoyage/wordDistribution/most_common_words.csv"):
    df = pd.read_csv(data)
    word_distribution = df[df['word'] == word].iloc[0].to_dict()
    ## only keep the score distribution
    word_distribution = {k: v for k, v in word_distribution.items() if k in ['1', '2', '3', '4', '5']}
    return word_distribution


score_distribution = {'1': 52268, '2': 29769, '3': 42640, '4': 80655, '5': 363122}
word = "!!!!!!!!!!!!!!!!"
scores_specific_word = ExtractSPecificWordDistribution(word)

# Normalize the data
scores_specific_word_normalized = NormalizeDistribution(scores_specific_word)
score_distribution_normalized = NormalizeDistribution(score_distribution)

# Sort keys to ensure consistent order
specific_values = list(scores_specific_word_normalized.values())
distribution_values = list(score_distribution_normalized.values())

# Plotting
plt.bar(
    [n - 0.4 for n in range(1, 6)],  # Shift specific word bars left
    specific_values,
    width=0.4,
    align="edge",
    label=f'Word "{word}"',
)

plt.bar(
    range(1, 6),  # All words bars start at score
    distribution_values,
    width=0.4,
    align='edge',
    label='Score Distribution',
)

plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title(f'Distribution of "{word}" compared with score distribution')
plt.legend()  # Auto-show labels from plt.bar() calls
plt.savefig(f"Nettoyage/wordDistribution/graphs/score_distribution_{word}.png")
plt.show()