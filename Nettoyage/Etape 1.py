#%%
# Etape 1, vérification des données (pas de données manquantes etc)
import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

df = pd.read_csv(r'C:\Users\fanny\OneDrive\Bureau\Ingé2\Projet\Reviews.csv')

print('Variable names:', *df.columns)
print(df.isnull().sum())
des = df.describe()
print(des)
print(df.info())


#%%
# Test sur une seule phrase

df = pd.read_csv("Reviews.csv", delimiter=',', quoting=csv.QUOTE_ALL)
text = df['Text'][0]

stop_words = set(stopwords.words('english')) 
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 


print("\n\nOriginal Sentence \n\n")
print(" ".join(word_tokens)) 

print("\n\nFiltered Sentence \n\n")
print(" ".join(filtered_sentence)) 


#%%

# enlever tous les stopwords et les < br />
with open('Reviews_without_stopwords_br_I.csv', 'w', encoding="utf-8", newline='') as csvfile:

    stop_words = set(stopwords.words('english'))

    df = pd.read_csv("Reviews.csv", delimiter=',', quoting=csv.QUOTE_ALL)
    text_list = df['Text']

    for i, line in enumerate(text_list):
        word_tokens = word_tokenize(line)
        filtered_sentence = [] 
        
        for word in word_tokens:
            if (word not in stop_words) and (word not in ['<', '/', 'br', '>', 'I']) :
                
                filtered_sentence.append(word)
        filtered_sentence = " ".join(filtered_sentence)
        csvfile.write(filtered_sentence + '\n')

    
#%%

# test pour enlever <br/>
text = "Product received advertised. < br / > < br / > < href= '' http : //www.amazon.com/gp/product/B001GVISJM '' > Twizzlers , Strawberry , 16-Ounce Bags ( Pack 6 ) < /a >"

word_tokens = word_tokenize(text)
filtered_sentence = [] 

for word in word_tokens:
    if (word not in stop_words) and word not in ['<', '/', 'br', '>'] :
        
        filtered_sentence.append(word)
filtered_sentence = " ".join(filtered_sentence)


print(filtered_sentence)
        
#%%

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

df = pd.read_csv(r'C:\Users\fanny\OneDrive\Bureau\Ingé2\Projet\Reviews_without_stopwords_br_I.csv', sep='delimiter', header=None)

text = " ".join(df[0])

wc = WordCloud(width = 300, height = 300, stopwords=[], background_color='white').generate(text)

# Remove the axis and display the data as image
plt.axis("off")
plt.imshow(wc, interpolation = "bilinear")


# plt.show() 
