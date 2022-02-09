import pandas as pd
import numpy as np

import pickle as pk

import warnings
warnings.filterwarnings("ignore")
from bs4 import BeautifulSoup
import unicodedata
import re

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist


def remove_html_tags_func(text):
    '''
    Removes HTML-Tags from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without HTML-Tags
    ''' 
    return BeautifulSoup(text, 'html.parser').get_text()



def remove_url_func(text):
    '''
    Removes URL addresses from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without URL addresses
    ''' 
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_accented_chars_func(text):
    '''
    Removes all accented characters from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without accented characters
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

def remove_irr_char_func(text):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without irrelevant characters
    '''
    return re.sub(r'[^a-zA-Z]', ' ', text)

def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without extra whitespaces
    ''' 
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def word_count_func(text):
    '''
    Counts words within a string
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    return len(text.split())

# from contractions import CONTRACTION_MAP 
# import re 
    
# def expand_contractions(text, map=CONTRACTION_MAP):
#     pattern = re.compile('({})'.format('|'.join(map.keys())), flags=re.IGNORECASE|re.DOTALL)
#     def get_match(contraction):
#         match = contraction.group(0)
#         first_char = match[0]
#         expanded = map.get(match) if map.get(match) else map.get(match.lower())
#         expanded = first_char+expanded[1:]
#         return expanded     
#     new_text = pattern.sub(get_match, text)
#     new_text = re.sub("'", "", new_text)
#     return new_text


# from pycontractions import Contractions
# cont = Contractions(kv_model=model)
# cont.load_models()# 

# def expand_contractions(text):
#     text = list(cont.expand_texts([text], precise=True))[0]
#     return text

def token_data(df, text_col_name, nex_col_prefix="", new_col_prefix="", token_name=None):
    df_token = df.copy()

    tokenizer = nltk.RegexpTokenizer(r'\w+')
    
    if token_name is None:
        token_name = nex_col_prefix+"tokens"+new_col_prefix
        df_token[token_name] = df_token[text_col_name].apply(lambda x: tokenizer.tokenize(x.lower()))  
    
    # on récupère la fréquence totale de chaque mot
    freq_name = nex_col_prefix+"freq"+new_col_prefix
    stat_name = nex_col_prefix+"word_count"+new_col_prefix
    word_unique_name = nex_col_prefix+"unique_word_count"+new_col_prefix

    df_token[stat_name] = df_token[token_name].apply(lambda x: len(x))
    df_token[freq_name] = df_token[token_name].apply(lambda x: nltk.FreqDist(x))
    df_token[word_unique_name] = df_token[freq_name].apply(lambda x: len(x.keys()))
    df_token = df_token.sort_values(by=[word_unique_name, stat_name], ascending=False)
    return df_token

def df_word_tokenize(df, text_col_name, nex_col_prefix="", new_col_prefix=""):
    df_token = df.copy()

    token_name = nex_col_prefix+"word_tokenize"+new_col_prefix

    df_token[token_name] = df_token[text_col_name].apply(lambda x: word_tokenize(x.lower()))
    return df_token

def remove_english_stopwords_func(text):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without Stop Words
    ''' 
    return remove_stopwords_func(text, language="english")


def remove_french_stopwords_func(text):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without Stop Words
    ''' 
    return remove_stopwords_func(text, language="french")

def remove_stopwords_func(text, language="french"):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
        language (str, optionnal) : french, english, ... default : french
    
    Returns:
        Clean string without Stop Words
    ''' 
    # check in lowercase 
    t = [token for token in text if token.lower() not in stopwords.words(language)]
    text = ' '.join(t)    
    return text