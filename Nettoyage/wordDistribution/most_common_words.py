#most common words
import pandas as pd
import matplotlib.pyplot as plt

n_words = 50

df = pd.read_csv("Reviews.csv")

# value_counts()
text_list = df['Text']
score_list = df['Score']
most_common_words = {}


for i, text in enumerate(text_list):
    words = text.split()
    for word in words:
        if word in most_common_words:
            most_common_words[word]["count"] += 1
            most_common_words[word]["score"][score_list[i]] += 1
        else:
            most_common_words[word] = {"count": 1, "score": {score_list[i]: 1}}

# sort the dictionary by value
sorted_words = sorted(most_common_words.items(), key=lambda x: x[1], reverse=True)

output_csv = pd.DataFrame(sorted_words[:n_words], columns=['Word', 'Frequency'])
output_csv.to_csv('most_common_words.csv', index=False)

# plot the most common words
plt.bar(range(n_words), [x[1] for x in sorted_words[:n_words]], align='center', color=)
plt.xticks(range(n_words), [x[0] for x in sorted_words[:n_words]], rotation=90)
plt.show()