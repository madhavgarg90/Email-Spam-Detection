# 📧 Spam Detection using NLP and Machine Learning

This project implements a **Spam Detection System** using Natural Language Processing (NLP) and compares the performance of three different classification models:

1. ✅ Multinomial Naïve Bayes  
2. 🌳 Decision Tree Classifier (Custom-built)
3. 🌲 Random Forest Classifier (Custom-built)  

We use the SMS Spam Collection Dataset for training and evaluating the models.

---

## 🗂️ Dataset

**Dataset Used:** SMS Spam Collection  
- Total Entries: **5,572**  
- Columns: `label` (`ham` or `spam`), `message`  
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 
- ![Head](Image/head.png)

---

## 🛠️ Tech Stack

- Python 
- Pandas, NumPy  
- NLTK for NLP preprocessing  
- Scikit-learn for Naïve Bayes  
- Custom implementation of Decision Tree & Random Forest  
- Matplotlib & Seaborn for visualization  
- Google Colab for notebook execution  
- Pickle for model persistence  

---

## 🧼 Text Preprocessing Pipeline

1. Remove non-alphabet characters using regex  
2. Convert to lowercase  
3. Tokenize the message  
4. Remove stopwords (using `nltk.corpus.stopwords`)  
5. Apply stemming (`PorterStemmer`)  
6. Vectorize using `CountVectorizer(max_features=4000)`  

---

## 🤖 Models Used

| Model                    | Type         | Library        |
|--------------------------|--------------|----------------|
| Multinomial Naïve Bayes  | Built-in     | `sklearn`      |
| Decision Tree            | From Scratch | Custom Logic   |
| Random Forest            | From Scratch | Custom Logic   |

---

## 📈 Model Performance

| Model                    | Accuracy | 
|--------------------------|----------|
| Multinomial Naïve Bayes  | 98.2%    | 
| Decision Tree Classifier | 95.7%    | 
| Random Forest            | 86.5%    | 

**🏆 Best Model: Multinomial Naïve Bayes**

---

## 💾 Model Saving

All trained models are saved using `pickle`:

- `RFC.pkl` – Random Forest  
- `DTC.pkl` – Decision Tree  
- `MNB.pkl` – Multinomial Naïve Bayes  

---

## 🔍 Example Prediction

![Example Prediction](Image/prediction.png)



## 🚀 Run Instructions

```bash
# Install required libraries
pip install numpy pandas matplotlib nltk scikit-learn
```

Run the notebook in Google Colab or locally using Jupyter.



---
