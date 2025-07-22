# 📧 Spam Detection using NLP and Machine Learning

This project implements a **Spam Detection System** using Natural Language Processing (NLP) and compares the performance of three different classification models:

1. ✅ Multinomial Naïve Bayes  
2. 🌳 Decision Tree Classifier  
3. 🌲 Random Forest Classifier (Custom-built)  

We use the SMS Spam Collection Dataset for training and evaluating the models.

---

## 🗂️ Dataset

**Dataset Used:** SMS Spam Collection  
- Total Entries: **5,572**  
- Columns: `label` (`ham` or `spam`), `message`  
- Source: [Kaggle – SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  

---

## 🛠️ Tech Stack

- Python 🐍  
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

| Model                    | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|--------------------------|----------|------------------|---------------|-----------------|
| Multinomial Naïve Bayes  | 98.2%    | 0.93             | 0.93          | 0.93            |
| Decision Tree Classifier | 95.7%    | 0.97             | 0.71          | 0.82            |
| Random Forest (Custom)   | 86.5%    | 1.00             | 0.00          | 0.00            |

**🏆 Best Model: Multinomial Naïve Bayes**

---

## 🔍 Example Prediction

```python
predict_spam_or_ham("Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.")
