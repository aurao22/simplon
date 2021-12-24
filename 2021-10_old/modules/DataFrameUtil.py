import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import inspect
import pandas as pd
import ListeUtil as myList
import inspect

def displayInfo(df, verbose=False):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("shape:", df.shape)
    print("line:", df.index)
    print("Columns ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(df.columns)
    print("Describe    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(df.describe())
    # print("Describe (all) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(df.describe(include="all"))
    print("Info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(df.info())
    if verbose:
        print("Types ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(df.dtypes)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def cleanDuplicated(df, idColumnName, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    df = df.copy()
    # Existe-t-il des doublons sur l'ID dans ce jeu de données ?
    if verbose:
        print("\n", df.shape, df[idColumnName].nunique())
        print(df[idColumnName].duplicated().sum())

    # on compte le nombre de valeurs manquantes pour la ligne et on stocke dans une nouvelle colonne
    df['NB_NAN'] = df.isna().sum(axis=1)
    # trie des lignes en fonction du nombre de valeurs manquantes
    df = df.sort_values('NB_NAN')
    if verbose:
        print(df['NB_NAN'])
    # suppression des duplicatas en gardant les versions les mieux remplies
    df = df.drop_duplicates(idColumnName, keep='first')
    # on supprime la colonne qui n'est plus utile
    df = df.drop('NB_NAN', axis=1)
    print("......................................END")
    if verbose:
        print(df.shape, df[idColumnName].nunique())
        print(df[idColumnName].duplicated().sum())
    return df


def displayMissingValues(df, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    # Existe-t-il des valeurs manquantes dans ce jeu de données ?
    nadatas = df.isna().sum()
    print("......................................END")
    if verbose == True:
        print(nadatas)


def addColumnNb_NAN(df, newColumnName='NB_NAN', verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    df = df.copy()
    # on compte le nombre de valeurs manquantes pour la ligne et on stocke dans une nouvelle colonne
    df[newColumnName] = df.isna().sum(axis=1)
    # trie des lignes en fonction du nombre de valeurs manquantes
    df = df.sort_values(newColumnName)
    print("......................................END")
    if verbose:
        print(df[newColumnName])
    return df


def cleanDuplicatedData(df, idColumnName, newColumnName='NB_NAN', verbose=False, removeNbNanCol=True):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    df = addColumnNb_NAN(df, newColumnName, verbose)
    # suppression des duplicatas en gardant les versions les mieux remplies
    df = df.drop_duplicates(idColumnName, keep='first')
    if removeNbNanCol:
        # on supprime la colonne qui n'est plus utile
        df = df.drop(newColumnName, axis=1)
    if verbose:
        print("Function", functionName, end='')
    print("......................................END")
    return df


def replaceNaNByMedianValue(df, columName, verbose=False):
    """Remplace les valeurs NaN par la valeur mediane de la colonne
                    Args:
                        df (DataFrame): le dataFrame à traiter
                        columName (String) : Nom de la colonne
                        verbose (True or False): True pour mode debug
                    Returns:
                        DataFrame : an updated DataFrame
                    """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    if verbose:
        print("Function", functionName)
    else:
        print("Function", functionName, end='')
    res = df.copy()
    median = res[columName].median()
    res[columName].fillna(value=median, inplace=True)
    if verbose:
        print(median)
        print(res[columName])
        print(res[columName].isna().sum())

    myList.logEnd("Function " + functionName, verbose)
    return res




