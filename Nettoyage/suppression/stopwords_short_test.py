#%%
# Etape 1, vérification des données (pas de données manquantes etc)
import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

#%%

# test
with open('../datasets/Reviews_copy.csv', 'w', encoding="utf-8", newline='') as reviews_original:

    stop_words = pd.read_csv("wordDistribution/stopwords.txt")

    reviews_without_stopword = pd.read_csv("../datasets/Reviews_without_stopwords_br.csv", quoting=csv.QUOTE_ALL)
    text_list = reviews_without_stopword['Text']

    for i, line in enumerate(text_list):
        
        reviews_original.write(line)

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
# Test sur une seule phrase

df = pd.read_csv('../datasets/Reviews_copy.csv', delimiter=',', quoting=csv.QUOTE_ALL)
text = df['Text'][0]

stop_words = pd.read_csv("../Nettoyage/wordDistribution/stopwords.txt")
#stop_words = stopwords.txt
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

 #df.drop(['Text'], axis=1)
#df.insert(0, 'Text without stopwords', [filtered_sentence])
#df.__setitem__('Text without stopwords', filtered_sentence)
#df.iloc[0]['Text without stopwords'] = filtered_sentence

print("\n\nOriginal Sentence \n\n")
print(" ".join(word_tokens)) 

print("\n\nFiltered Sentence \n\n")
print(" ".join(filtered_sentence)) 
