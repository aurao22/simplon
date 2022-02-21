



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
