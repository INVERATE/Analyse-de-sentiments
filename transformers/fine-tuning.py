import pandas as pd

# Chargement du fichier CSV
df = pd.read_csv("datasets/Reviews_clean_lemmatized_medium.csv")  # Correction de "daatasets" -> "datasets"

# Texte = Text_without_stopwords, Label = Score - 1 (pour avoir des classes 0 à 4)
df['label'] = df['Score'] - 1
df = df[['Text_without_stopwords', 'label']]
df = df.rename(columns={'Text_without_stopwords': 'text'})


from datasets import Dataset

dataset = Dataset.from_pandas(df)

# Split en train/test
dataset = dataset.train_test_split(test_size=0.2, seed=42)


from transformers import AutoTokenizer

model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)


from transformers import AutoModelForSequenceClassification

# Nombre de labels (ici 5 classes : score de 0 à 4 après transformation)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=5)


import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

# Assurez-vous que les données sont dans le bon format pour l'entraînement
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

trainer.train()

# Évaluation du modèle
results = trainer.evaluate()
print(f"Résultats de l'évaluation: {results}")

# Sauvegarde du modèle et du tokenizer
model.save_pretrained("./fine_tuned_roberta")
tokenizer.save_pretrained("./fine_tuned_roberta")