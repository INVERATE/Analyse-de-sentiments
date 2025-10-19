# 🧠 Sentiment Analysis of Amazon Reviews using various Models

This project focuses on predicting customer sentiment from text reviews using **Machine Learning**, **Deep Learning**, and **Transformers**.  
It was developed as part of an academic project at **ESME (Artificial Intelligence major)** by **Néo Colpin** and **Fanny Badoulès**, under the supervision of **Wajd Meskini**.

---

## 📋 Project Overview

A restaurant chain collects online customer feedback but **does not allow users to rate products directly**.  
Our goal was to build a model capable of **predicting the sentiment (1–5 stars)** from text reviews only, lacking all the other informations from the customers.

### 🎯 Objectives
1. Design a model that predicts the sentiment of a comment.  
2. Apply **transfer learning** from an external dataset.  
3. Deploy the final model as an **API** for web integration.  
4. Develop an **Android app** to facilitate access to the service.

---

## 🧩 Dataset

- **Source:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  
- Contains thousands of customer reviews with ratings from 1 to 5.  

### 🧹 Preprocessing Steps
- Text cleaning and lemmatization  
- Stopwords removal  
- TF-IDF matrix creation for Machine Learning
- Word embeddings with **GloVe** for Deep Learning models
![stopwords](https://github.com/INVERATE/Analyse-de-sentiments/blob/main/images/Capture%20d'%C3%A9cran%202025-10-19%20233933.png?raw=true)
---

## ⚙️ Methods

| Approach | Description |
|-----------|--------------|
| **Machine Learning** | Logistic Regression, Decision Trees, SVM — optimized using GridSearchCV |
| **Deep Learning** | Feedforward and Recurrent (LSTM) models — with and without word embeddings |
| **Transformers** | **RoBERTa** (pretrained) and **RoBERTa fine-tuned** on our dataset |

---

## 📈 Results

| Model | Training Time | F1-Score (Avg.) | Observations |
|--------|----------------|-----------------|---------------|
| Logistic Regression | 14s | Low (biased toward score 5) | Poor nuance detection |
| LSTM | 1h | Moderate | Improved accuracy, high cost |
| **RoBERTa (Fine-tuned)** | **30min** | **Best** | Balanced, fast, and robust |

**RoBERTa fine-tuned achieved the best results**, reducing classification errors while maintaining reasonable training time.
![Results](https://github.com/INVERATE/Analyse-de-sentiments/blob/main/images/Capture%20d'%C3%A9cran%202025-10-19%20234041.png?raw=true)

---

## 🚀 Deployment

- A [**REST API**](https://projet-amazon.onrender.com/) was developed to allow merchants to submit new reviews and get automatic sentiment predictions.  
- An **Android application** created with Flutter provides a user-friendly interface for accessing the model’s predictions.
![app](https://github.com/INVERATE/Analyse-de-sentiments/blob/main/images/Screenshot_2025-05-26-11-38-21-43_5085926b8b79a82b5e480693d46976f9.jpg?raw=true)
---
