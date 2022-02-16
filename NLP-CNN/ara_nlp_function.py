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

def words_by_weight(words_dic):
    res_dic = {}
    count_keys = sorted(words_dic.values(), reverse=True)
    for k in count_keys:
        res_dic[k] = []

    for key, v in words_dic.items():
        res_dic[v].append(key)
    
    return res_dic


def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    # pourrait être remplacé par : string.punctuation ?
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
        text (str or list[str] ): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    if isinstance(text, str):
        return len(text.split())
    elif isinstance(text, list):
        return word_count_func(' '.join(text))
    return len(text.split())


def norm_stemming_func(text):
    '''
    Stemming tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use PorterStemmer() to stem the created tokens
    
    Args:
        text (str or list[str] ): String to which the functions are to be applied, string
    
    Returns:
        str or list[str] with stemmed words
    ''' 
    words = None 
    if isinstance(text, str):
        words = word_tokenize(text)
        text = ' '.join([PorterStemmer().stem(word) for word in words])
    elif isinstance(text, list):
        words = text 
        text = [PorterStemmer().stem(word) for word in words]
    return text




def norm_lemm_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
    
    Args:
        text (str or list[str] ): String to which the functions are to be applied, string
    
    Returns:
        str or list[str] with lemmatized words
    '''  
    words = None 
    if isinstance(text, str):
        words = word_tokenize(text)
        text = ' '.join([WordNetLemmatizer().lemmatize(word) for word in words])
    elif isinstance(text, list):
        words = text 
        text = [WordNetLemmatizer().lemmatize(word) for word in words]
    return text

def norm_lemm_v_func(text):
    '''
    Lemmatize tokens from string 
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is set to 'v' for verb
    
    Args:
        text (str or list[str] ): String to which the functions are to be applied, string
    
    Returns:
        str or list[str] with lemmatized words
    '''  
    words = None 
    if isinstance(text, str):
        words = word_tokenize(text)
        text = ' '.join([WordNetLemmatizer().lemmatize(word, pos='v') for word in words])
    elif isinstance(text, list):
        words = text
        text = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
    return text


def norm_lemm_a_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is set to 'a' for adjective
    
    Args:
        text (str or list[str] ): String to which the functions are to be applied, string
    
    Returns:
        str or list[str] with lemmatized words
    ''' 
    words = None 
    if isinstance(text, str):
        words = word_tokenize(text)
        text = ' '.join([WordNetLemmatizer().lemmatize(word, pos='a') for word in words])
    elif isinstance(text, list):
        words = text
        text = [WordNetLemmatizer().lemmatize(word, pos='a') for word in words]
    return text


def get_wordnet_pos_func(word):
    '''
    Maps the respective POS tag of a word to the format accepted by the lemmatizer of wordnet
    
    Args:
        word (str or list): Word to which the function is to be applied, string
    
    Returns:
        POS tag or list[POS tag], readable for the lemmatizer of wordnet
    '''
    if isinstance(word, str):     
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    elif isinstance(word, list):
        res = []
        for w in word:
            r = get_wordnet_pos_func(w)
            res.append(r)
        return res

def norm_lemm_POS_tag_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is determined with the help of function get_wordnet_pos()
    
    Args:
        text (str or list[str] ): String to which the functions are to be applied, string
    
    Returns:
        str or list[str] with lemmatized words
    ''' 
    if isinstance(text, str):   
        words = word_tokenize(text)
        text = ' '.join([WordNetLemmatizer().lemmatize(word, get_wordnet_pos_func(word)) for word in words])
        return text
    elif isinstance(text, list):
        words = text
        text = [WordNetLemmatizer().lemmatize(word, get_wordnet_pos_func(word)) for word in words]
    return text


def norm_lemm_v_a_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() with POS tag 'v' to lemmatize the created tokens
    Step 3: Use word_tokenize() to get tokens from generated string        
    Step 4: Use WordNetLemmatizer() with POS tag 'a' to lemmatize the created tokens
    
    Args:
        text (str or list[str] ): String to which the functions are to be applied, string
    
    Returns:
        str or list[str] with lemmatized words
    '''
    if isinstance(text, str):   
        words1 = word_tokenize(text)
        words2 = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words1]
        text2 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='a') for word in words2])
        return text2
    elif isinstance(text, list):
        words1 = text
        words2 = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words1]
        text2 = [WordNetLemmatizer().lemmatize(word, pos='a') for word in words2]
        return text2

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
        text (str or list): String to which the function is to be applied, string
        language (str, optionnal) : french, english, ... default : french
    
    Returns:
        Clean (str or list) without Stop Words
    ''' 
    # check in lowercase 
    if isinstance(text, str):
        res_list = remove_stopwords_func(text.split(" "), language=language)
        text = ' '.join(res_list)    
        return text
    elif isinstance(text, list):
        t = [token for token in text if token.lower() not in stopwords.words(language)]
        return t


def test_remove_stopwords_func():
    list_text = ['cnn',  'rory',  'mcilroy',  'is',  'off',  'to',  'a',  'good',
                'start',  'at',   'the',  'scottish', 'open', 'he', 's', 'hoping', 'for', 'a', 'good', 'finish',
                'too', 'after', 'missing', 'the', 'cut', 'at', 'the', 'irish', 'open', 'mcilroy', 'shot', 'a',
                'course', 'record', '7', 'under', 'par', '64', 'at', 'royal', 'aberdeen', 'on', 'thursday', 'and',
                'he', 'was', 'actually', 'the','second', 'player', 'to', 'better', 'the', 'old', 'mark', 'sweden',
                's', 'kristoffer', 'broberg', 'had', 'earlier', 'fired', 'a', '65', 'mcilroy', 'carded', 'eight',
                'birdies', 'and', 'one', 'bogey', 'in', 'windy', 'chilly', 'conditions', 'going', 'out', 'this',
                'morning', 'in', 'these', 'conditions', 'i', 'thought', 'anything', 'in', 'the', '60s', 'would',
                'be', 'a', 'good', 'score', 'so', 'to', 'shoot', 'something', 'better', 'than', 'that', 'is',
                'pleasing', 'mcilroy', 'was', 'quoted', 'as', 'saying', 'by', 'the', 'european', 'tour', 's',
                'website', 'a', 'win', 'sunday', 'would', 'be', 'the', 'perfect', 'way', 'for', 'former', 'no',
                '1', 'mcilroy', 'to', 'prepare', 'for', 'the', 'british', 'open', 'which', 'starts', 'next', 'week',
                'at', 'royal', 'liverpool', 'he', 'won', 'the', 'last', 'of', 'his','two', 'majors', 'in', '2012', 'everything',
                'was', 'pretty', 'much', 'on', 'mcilroy', 'said', 'i', 'controlled', 'my', 'ball', 'flight', 'really', 'well',
                'which', 'is', 'the', 'key', 'to', 'me', 'playing', 'well', 'in', 'these', 'conditions', 'and', 'on',
                'these', 'courses', 'i', 've', 'been', 'working', 'the', 'last', '10', 'days', 'on', 'keeping', 'the',
                'ball', 'down', 'hitting', 'easy', 'shots', 'and', 'taking', 'spin', 'off', 'it', 'and', 'i', 'went', 'out',
                'there', 'today', 'and', 'really', 'trusted', 'what', 'i', 'practiced', 'last', 'year', 'phil', 'mickelson', 'used',
                'the', 'scottish', 'open', 'at', 'castle', 'stuart', 'as', 'the', 'springboard', 'to', 'his', 'british', 'open',
                'title', 'and', 'his', '68', 'leaves', 'him', 'well', 'within', 'touching', 'distance', 'of', 'mcilroy', 'mickelson',
                'needs', 'a', 'jolt', 'of', 'confidence', 'given', 'that', 'lefty','has', 'slipped', 'outside', 'the', 'top',
                '10', 'in', 'the', 'rankings', 'and', 'hasn', 't', 'finished', 'in', 'the', 'top', '10', 'on', 'the',
                'pga', 'tour', 'this', 'season', 'i', 'thought', 'it', 'was', 'tough', 'conditions', 'mickelson', 'said',
                'in', 'an', 'audio', 'interview', 'posted', 'on', 'the', 'european', 'tour', 's', 'website', 'i', 'was',
                'surprised', 'to', 'see', 'some', 'low', 'scores', 'out', 'there', 'because', 'it', 'didn', 't', 'seem', 'like',
                'it', 'was', 'playing', 'easy', 'and', 'the', 'wind', 'was', 'pretty', 'strong', 'i', 'felt', 'like', 'i', 'played', 'well',
                'and', 'had', 'a', 'good', 'putting', 'day', 'it', 'was', 'a', 'good', 'day', 'last', 'year', 's', 'u', 's', 'open', 'champion',
                'justin', 'rose', 'was', 'tied', 'for', '13th', 'with', 'a', '69', 'but', 'jonas', 'blixt', 'who', 'tied', 'for', 'second', 'at',
                'the', 'masters', 'was', 'well', 'adrift', 'following', 'a', '74']
    
    res_list = remove_english_stopwords_func(list_text)
    print(res_list)

    text = 'cnn rory mcilroy is off to a good start at the scottish open he hoping for a good finish too after missing the cut at the irish open mcilroy shot a course record 7 under 64 at royal aberdeen thursday and he was actually the second player to better the old mark sweden kristoffer broberg had earlier fired a 65 mcilroy carded eight birdies and one bogey in windy chilly conditions going out this morning in these conditions i thought anything in the 60s would be a good score so to shoot something better than that is pleasing mcilroy was quoted saying by the european tour website a win sunday would be the perfect way for former no 1 mcilroy to prepare for the british open which starts next week at royal liverpool he won the last of his two majors in 2012 everything was pretty much mcilroy said i controlled my ball flight really well which is the key to playing well in these conditions and these courses i ve been working the last 10 days keeping the ball down hitting easy shots and taking spin off it and i went out there today and really trusted what i practiced last year phil mickelson used the scottish open at castle stuart the springboard to his british open title and his 68 leaves him well within touching distance of mcilroy mickelson needs a jolt of confidence given that lefty has slipped outside the top 10 in the rankings and hasn finished in the top 10 the pga tour this season i thought it was tough conditions mickelson said in an audio interview posted the european tour website i was surprised to see some low scores out there because it didn seem like it was playing easy and the wind was pretty strong i felt like i played well and had a good putting day it was a good day last year u open champion justin rose was tied for 13th with a 69 but jonas blixt who tied for second at the masters was well adrift following a 74'
    res_txt = remove_english_stopwords_func(text)
    print(res_txt)


if __name__ == "__main__":
    test_remove_stopwords_func()
    test ={'march': 4, '14': 11, 'be': 41, 'my': 1}
    print(words_by_weight(test))
