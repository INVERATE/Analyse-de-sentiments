import pickle

# Utilisation du meilleur modèle pour prédire un commentaire
with open('Machine Learning/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('Machine Learning/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

sentence = "this is not a good product"
sentence_vector = tfidf.transform([sentence])
predicted_score = best_model.predict(sentence_vector)

print(f"Predicted score : {predicted_score}")