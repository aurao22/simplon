# import urllib
# import bs4
# import re
import os
import nltk
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
nltk.download('stopwords','wordnet','omw-1.4','popular')
from nltk import word_tokenize,WordPunctTokenizer,ngrams
from nltk.stem import WordNetLemmatizer,PorterStemmer 
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
from wordcloud import WordCloud, STOPWORDS as wc_stopwords, ImageColorGenerator
from nltk.corpus import stopwords as nltk_stopwords
import matplotlib.pyplot as plt
import contractions
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder



def clean(txt,x=1):
    t= contractions.fix(txt)
    l=tok1(t)
    l=only_word(l)
    l=lem(l,x)
    l=stop_w(l,x)
    return " ".join(l)





def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)




def tok1(txt):
    tokenizer = nltk.RegexpTokenizer(r'\w+') #\D
    t = tokenizer.tokenize(txt.lower())
    return t



def only_word(lst):
    l=[]
    for w in lst:
        if len(w)<3:
            pass
        # elif w.isnumeric():
        #     pass
        else:
            if not has_numbers(w):
                l.append(w)
    return l



def lem(lst,x=1):
    lemma=WordNetLemmatizer()

    if x==1:
        lem=[lemma.lemmatize(word)for word in lst]
    elif x==2:
        lem=[lemma.lemmatize(lemma.lemmatize(lemma.lemmatize(word), pos='v'),pos='a') for word in lst ]

    return lem




def stop_w(lst,x=1,add=[]):
    add_sw=[]

    with open("english_stopwords.txt")as f:
        git_stopwords=f.read()
    git_stopwords=git_stopwords.split('\n')  

    stopwords=set()
    if x==1 or x==2:
        stopwords.update(nltk_stopwords.words('english'))
    if x==2:
        stopwords.update(wc_stopwords)
        stopwords.update(git_stopwords)
    

    return [word for word in lst if word not in stopwords]


def tfidf(corpus):
    
    vectorizer = TfidfVectorizer(stop_words=stop_w_lst(), ngram_range= (1,1))
    X = vectorizer.fit_transform(corpus)

    df=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())

    tfidf_weights = [(word, X.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()] #calcule le poids des mots et les rend dans un tuple
    weight= sorted(tfidf_weights, key=lambda tup: tup[1], reverse=True)
    return weight,df



def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def stop_w_lst():
    add_sw=[]

    with open("english_stopwords.txt")as f:
        git_stopwords=f.read()
    git_stopwords=git_stopwords.split('\n')  

    stopwords=set()
    # stopwords.update(wc_stopwords)
    stopwords.update(nltk_stopwords.words('english'))
    # stopwords.update(add_sw)
    # stopwords.update(git_stopwords)

    return stopwords





def ohe(data,columns):
    '''OneHotEncoder
    Pour les colonne comportant des variables qulitative Nominal
    cad: qui n ont pas d ordre entre eux
    Va creer une colonne pour chaque valeur unique qui va etre composé de Booleen
    Supprim la colonne initiale
    data : df 
    column: str (nom de la colonne a encoder)'''
    
    for column in columns:
        result = OneHotEncoder().fit_transform(data.loc[:,column].values.reshape(-1, 1)).toarray()

        # recupere les valeur unique de la colonne pour en faire des nom de colonne (nom de colonne + val)
        tab=data[column].unique()
        tab.sort()
        tab2=[]
        for i in range(len(tab)):
            tab2.append(column +" / "+ str(tab[i]))

        # Appending columns
        data[tab2]=pd.DataFrame(result,index = data.index)
        data.drop(columns=column,inplace=True)
        
    return data

























