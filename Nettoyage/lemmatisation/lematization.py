# %%

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# Télécharger les ressources nécessaires
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")
# nltk.download("omw-1.4")
#nltk.download("punkt_tab")

lemmatizer = WordNetLemmatizer()


# Fonction pour convertir les POS tags en format WordNet
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Fonction de lemmatisation d’un texte entier
def lemmatize_text(text):
    if pd.isna(text):
        return ""
    
    # Tokenisation et lemmatisation
    text = text.lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens) # attribut une fonction grammaticale à chaque mot
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged
    ]
    return " ".join(lemmatized)


# Lire les données
df = pd.read_csv("datasets/Reviews_clean_short.csv")
nom_colonne = "Text_without_stopwords"

# Appliquer la lemmatisation à toute la colonne avec `.apply`
df[nom_colonne] = df[nom_colonne].apply(lemmatize_text)

# Sauvegarder les résultats
df.to_csv("datasets/Reviews_clean_lemmatized_short.csv", index=False)