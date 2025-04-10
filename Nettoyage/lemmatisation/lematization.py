#%%

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Télécharger les ressources nécessaires

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")

# Fonction pour convertir les tags de NLTK en tags compatibles avec WordNet
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)

def NormalizeText(text):
    # Normaliser le texte (par exemple, mettre en minuscules, retirer la ponctuation, etc.)
    text = text.lower()
    return text


# Lire les données
df = pd.read_csv("./../../datasets/Reviews_clean_copy_short.csv")
nom_colonne = "Text_without_stopwords"
texts = df[nom_colonne]


lemmatizer = WordNetLemmatizer()
filtered_sentence = []
with open('./../../datasets/Reviews_lemmatized.csv', 'w', encoding="utf-8", newline='') as csvfile:

    # Lemmatiser les mots en utilisant le bon tag de pos
    for text in texts:
        text_normalized = NormalizeText(text)
        for word in text_normalized.split():
            wordnet_pos = get_wordnet_pos(word)  # Obtenir le bon tag pour lemmatize
            lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)

            filtered_sentence += word

        filtered_sentence = " ".join(filtered_sentence)
        csvfile.write(filtered_sentence + '\n')

            #df.loc[df[nom_colonne] == word, nom_colonne] = lemmatized_word

# Enregistrer le DataFrame mis à jour
#df.to_csv("Nettoyage/wordDistribution/most_common_words_lemmatized.csv", index=False)



#%%
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

# Create a sample dataframe
df = pd.read_csv("./../../datasets/Reviews_clean_copy_short.csv")

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Define a function to lemmatize text
def lemmatize_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Join the lemmatized words back into a string
    lemmatized_text = ' '.join(lemmatized_words)
    print(lemmatized_text)
    return lemmatized_text

# Apply the lemmatize_text function to the text column of the dataframe

df['Lemmatized_text'] = df['Text_without_stopwords'].apply(lemmatize_text)

