import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def del_stopwords(corpus,stopwords):
    corpus_without_stopwords = []
    for word in corpus:
        chiffre_in_word = False
        a = word.lower()
        for lettre in word:
            if lettre in chiffres:
                chiffre_in_word = True
        if a not in stopwords and len(word)>2 and chiffre_in_word == False:
            corpus_without_stopwords.append(a)
    return corpus_without_stopwords

stop_words = set(stopwords.words('french'))

chiffres = {"0","1","2","3","4","5","6","7","8","9"}



f = open(r'C:\Users\User\WORK\workspace-ia\PROJETS\thomas_presse\article_test.txt', 'r')

article_test = f.read()

f.close

tokenizer = RegexpTokenizer(r'\w+')

article_test = tokenizer.tokenize(article_test)
article_test = del_stopwords(article_test,stop_words)

var_int = ""
for word in article_test:
    var_int += word + " "
article_test = var_int

vectorizer_model = pickle.load(open(r'C:\Users\User\WORK\workspace-ia\PROJETS\thomas_presse\vectorizer_model.sav', 'rb'))

clf = pickle.load(open(r'C:\Users\User\WORK\workspace-ia\PROJETS\thomas_presse\LogisticRegression.sav', 'rb'))

X_test2 = vectorizer_model.transform([article_test])

df_test = pd.DataFrame(X_test2.toarray(),columns=vectorizer_model.get_feature_names())

pred = clf.predict(df_test)[0]

proba = round(np.max(clf.predict_proba(df_test)),2)

print(f"L'article provient du journal {pred} avec une probabilité de {proba}")

input("Vous pouvez quittez le programme")