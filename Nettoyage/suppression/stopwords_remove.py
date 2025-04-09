#%%
# Etape 1, vérification des données (pas de données manquantes etc)
import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
#%%
df = pd.read_csv('./../../datasets/Reviews.csv')

print('Variable names:', *df.columns)
print(df.isnull().sum())
des = df.describe()
print(des)
print(df.info())

#%%
with open('./../wordDistribution/stopwords_copy.txt', 'r+') as input_file:
    for line in input_file.readlines():
        line = line.strip()
        line = f'{line}. \n'
        input_file.write(line)


#%%

# enlever tous les stopwords et les < br />
with open('./../../datasets/Reviews_without_stopwords_br.csv', 'w', encoding="utf-8", newline='') as csvfile:

    stop_words = open("./../wordDistribution/stopwords_copy.txt",'r').read().split()

    df = pd.read_csv("./../../datasets/Reviews.csv", delimiter=',', quoting=csv.QUOTE_ALL)
    text_list = df['Text']
    
    for i, line in enumerate(text_list):
        word_tokens = word_tokenize(line)
        
        filtered_sentence = [] 
        
        for word in word_tokens:
            word = word.lower()
            if (word not in ['<', '/', 'br', '>',',', '``', "''",'.']) and (word not in stop_words) :
                
                filtered_sentence.append(word)
        filtered_sentence = " ".join(filtered_sentence)
        csvfile.write(filtered_sentence + '\n')

 #df.loc[i, 'Text'] = filtered_sentence
#%%
# ajouter la colonne 'Text without stopwords' dans reviews_copy

reviews_copy = pd.read_csv("./../../datasets/Reviews_copy.csv", delimiter=',', quoting=csv.QUOTE_ALL)
text_without_stopwords = pd.read_csv("./../../datasets/Reviews_without_stopwords_br.csv")
reviews_copy.drop('Text', axis=1, inplace=True)

reviews_copy['Text_without_stopwords']= text_without_stopwords['Text']
reviews_copy.to_csv('./../../datasets/Reviews_clean.csv')
#%% Map

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os

fields = ['Text_without_stopwords']
current_dir = os.getcwd()
print(current_dir)
df = pd.read_csv('./../../datasets/Reviews_clean.csv', usecols=fields)

text = df['Text_without_stopwords'].values 

wordcloud = WordCloud(background_color='white', width=800, height=600).generate(str(text))

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('./../wordDistribution/graphs/wordcloud_without_stopwords.png')
plt.show()


# %%
