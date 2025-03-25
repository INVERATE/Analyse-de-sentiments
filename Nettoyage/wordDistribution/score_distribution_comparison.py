import matplotlib.pyplot as plt

score_distribution = {1: 52268, 2: 29769, 3: 42640, 4: 80655, 5: 363122}
scores_specific_word = {5: 29205, 1: 5341, 3: 4605, 4: 8457, 2: 2990}
word = "1"

# Normalize the data
sum_specific = sum(scores_specific_word.values())
scores_specific_word = {k: v / sum_specific for k, v in scores_specific_word.items()}

sum_dist = sum(score_distribution.values())
score_distribution = {k: v / sum_dist for k, v in score_distribution.items()}

# Sort keys to ensure consistent order
sorted_scores = sorted(scores_specific_word.keys())
specific_values = [scores_specific_word[k] for k in sorted_scores]
distribution_values = [score_distribution[k] for k in sorted_scores]

# Plotting
plt.bar(
    [k - 0.4 for k in sorted_scores],  # Shift specific word bars left
    specific_values,
    width=0.4,
    align='edge',
    label=f'Word "{word}"'
)

plt.bar(
    sorted_scores,  # All words bars start at score
    distribution_values,
    width=0.4,
    align='edge',
    label='Score Distribution',
)

plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title(f'Distribution of "{word}" compared with score distribution')
plt.legend()  # Auto-show labels from plt.bar() calls
plt.savefig(f"distribution_{word}.png")
plt.show()