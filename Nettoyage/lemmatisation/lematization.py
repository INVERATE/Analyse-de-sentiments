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
df = pd.read_csv("datasets/Reviews.csv")
texts = df["Text"]
print(texts.head())

lemmatizer = WordNetLemmatizer()

# Lemmatiser les mots en utilisant le bon tag de pos
for text in texts:
    text_normalized = NormalizeText(text)
    for word in text.split():
        wordnet_pos = get_wordnet_pos(word)  # Obtenir le bon tag pour lemmatize
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        df.loc[df["word"] == word, "word"] = lemmatized_word

# Enregistrer le DataFrame mis à jour
df.to_csv("Nettoyage/wordDistribution/most_common_words_lemmatized.csv", index=False)
