# %%
import pandas as pd
import numpy as np

import nltk
import string
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
print('done')

# %%
nltk.download('punkt')
nltk.download('stopwords')

# %%
df = pd.read_csv("Emotion_Sentiment_DataSet.csv")

# %%
df.head(10)

# %%
df.shape

# %%
df.columns

# %%
df.isnull().sum()

# %%
df["Emotion"].value_counts()

# %%
df=df.drop(columns=["Unnamed: 0"])

# %%
df.head()

# %%
nlp = spacy.blank("en")  

# %%
texts = df["Text"].fillna("").astype(str)

docs = nlp.pipe(texts, batch_size=512)

df["tokens"] = [[token.text for token in doc] for doc in docs]

# %%


df.to_pickle("df_tokenized.pkl")

# %%
df["tokens"].head(10)

# %%
df["tokens"] = df["tokens"].apply(lambda x: [word.lower() for word in x])

# %%
df.to_pickle("step2_case_folding.pkl")

# %%

stop_words = set(stopwords.words("english"))

df["tokens"] = df["tokens"].apply(
    lambda x: [word for word in x if word not in stop_words]
)

# %%
df.to_pickle("step3_stop_word_removed.pkl")

# %%
stemmer = PorterStemmer()

df["tokens"] = df["tokens"].apply(
    lambda x: [stemmer.stem(word) for word in x]
)

# %%
df.to_pickle("step4_stemmer_removed.pkl")

# %%
nlp = spacy.load("en_core_web_sm")

df["lemmas"] = df["Text"].fillna("").apply(
    lambda text: [token.lemma_ for token in nlp(str(text))]
)

# %%
df.to_pickle("step5_stemmer_removed.pkl")

# %%
df["processed_text"] = df["lemmas"].apply(" ".join)

# %%
df.to_pickle("step6_process_text.pkl")

# %%
df.head()

# %%
X = df["processed_text"]
y = df["Emotion"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
bow = CountVectorizer()

X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

# %%

nb = MultinomialNB()

nb.fit(X_train_bow, y_train)

y_pred = nb.predict(X_test_bow)

# %%

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

# %%
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# %%
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_bow, y_train)

y_pred_lr_bow = lr.predict(X_test_bow)

# %%
print("========== Logistic Regression (BoW) ==========")

print("Accuracy :", accuracy_score(y_test, y_pred_lr_bow))
print("Precision:", precision_score(y_test, y_pred_lr_bow, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred_lr_bow, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred_lr_bow, average="weighted"))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_lr_bow))

print("\nClassification Report")
print(classification_report(y_test, y_pred_lr_bow))

# %%
svm = LinearSVC()

svm.fit(X_train_bow, y_train)

y_pred = svm.predict(X_test_bow)

# %%

print("========== Linear SVC (BoW) ==========")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# %%
tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# %%
nb_tfidf = MultinomialNB()

nb_tfidf.fit(X_train_tfidf, y_train)

y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)

# %%
print("Accuracy :", accuracy_score(y_test, y_pred_nb_tfidf))
print("Precision:", precision_score(y_test, y_pred_nb_tfidf, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred_nb_tfidf, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred_nb_tfidf, average="weighted"))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_nb_tfidf))

print("\nClassification Report")
print(classification_report(y_test, y_pred_nb_tfidf))

# %%

lr_tfidf = LogisticRegression(max_iter=1000)

lr_tfidf.fit(X_train_tfidf, y_train)

y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)

# %%
print("Accuracy :", accuracy_score(y_test, y_pred_lr_tfidf))
print("Precision:", precision_score(y_test, y_pred_lr_tfidf, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred_lr_tfidf, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred_lr_tfidf, average="weighted"))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_lr_tfidf))

print("\nClassification Report")
print(classification_report(y_test, y_pred_lr_tfidf))

# %%


svm_tfidf = LinearSVC()

svm_tfidf.fit(X_train_tfidf, y_train)

y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)

# %%
print("Accuracy :", accuracy_score(y_test, y_pred_svm_tfidf))
print("Precision:", precision_score(y_test, y_pred_svm_tfidf, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred_svm_tfidf, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred_svm_tfidf, average="weighted"))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_svm_tfidf))

print("\nClassification Report")
print(classification_report(y_test, y_pred_svm_tfidf))

# %%
import numpy as np
import pandas as pd

def show_all_model_prediction(text):
    
    models = {
        "Naive Bayes (BOW)": (nb, bow),
        "Logistic Regression (BOW)": (lr, bow),
        "SVM (BOW)": (svm, bow),
        "Naive Bayes (TF-IDF)": (nb_tfidf, tfidf),
        "Logistic Regression (TF-IDF)": (lr_tfidf, tfidf),
        "SVM (TF-IDF)": (svm_tfidf, tfidf)
    }

    print("="*70)
    print("Input Text:")
    print(text)
    print("="*70)

    for name, (model, vectorizer) in models.items():

        x = vectorizer.transform([text])

        prediction = model.predict(x)[0]

        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]

        else:
           
            scores = model.decision_function(x)[0]

            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()


        classes = model.classes_

        result = pd.DataFrame({
            "Class": classes,
            "Percentage": probs * 100
        })

        result = result.sort_values(
            by="Percentage",
            ascending=False
        )


        print("\nMODEL:", name)
        print("Prediction:", prediction)

        print(result.to_string(
            index=False,
            formatters={
                "Percentage": "{:.2f}%".format
            }
        ))

        print("-"*70)

# %%
%whos

# %%
import numpy as np
import pandas as pd


def show_all_model_prediction(text):

    models = {
        "Naive Bayes (BOW)": (nb, bow),
        "Logistic Regression (BOW)": (lr, bow),
        "SVM (BOW)": (svm, bow),

        "Naive Bayes (TF-IDF)": (nb_tfidf, tfidf),
        "Logistic Regression (TF-IDF)": (lr_tfidf, tfidf),
        "SVM (TF-IDF)": (svm_tfidf, tfidf)
    }


    print("="*70)
    print("Input Text:")
    print(text)
    print("="*70)


    for name, (model, vectorizer) in models.items():

        try:

            x = vectorizer.transform([text])

            prediction = model.predict(x)[0]


            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(x)[0]


            else:

                scores = model.decision_function(x)

               
                if len(scores.shape) > 1:
                    scores = scores[0]


                exp_scores = np.exp(
                    scores - np.max(scores)
                )

                probabilities = exp_scores / exp_scores.sum()



            classes = model.classes_


            result = pd.DataFrame({
                "Class": classes,
                "Confidence": probabilities * 100
            })


            result = result.sort_values(
                by="Confidence",
                ascending=False
            )


            print("\nMODEL:", name)
            print("Prediction:", prediction)

            print(
                result.to_string(
                    index=False,
                    formatters={
                        "Confidence": "{:.2f}%".format
                    }
                )
            )

            print("-"*70)


        except Exception as e:

            print("\nMODEL:", name)
            print("ERROR:", e)
            print("-"*70)

# %%
show_all_model_prediction(
     """

can't do anything in life
"""
)

# %%



