import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/Reviews.csv")

score_list = df['Score']

# plot the score distribution without sort
score_distribution = score_list.value_counts(sort=False).sort_index()
score_distribution.plot(kind='bar')
print(score_distribution)

output_csv = pd.DataFrame(score_distribution.items(), columns=['Score', 'Frequency'])
output_csv.to_csv('score_distribution.csv', columns=['Score', 'Frequency'], index=False)

plt.xlabel('Score')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.title('Score Distribution')
plt.show()