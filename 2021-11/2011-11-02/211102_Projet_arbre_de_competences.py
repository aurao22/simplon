import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import GraphiqueUtile as myGraph
import ListeUtil as myList
import seaborn as sns
import pandas as pd
import inspect


columnNameNotStudient = ['Code Rubrique', 'Rubrique', 'Code sous rubrique', 'Sous-Rubrique', 'Unité d’apprentissage ', "MEAN"]

fileName = "211102_Projet2Pierre-ARA-Complet.csv"

resFileName = "C:/Users/User/Documents/SIMPLON/workspace/mois1/semaine3/211102_Resultat.csv"

verbose = False


def renameIndex(df, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    df = df.copy()
    df = df.rename(index=lambda s: "U" + str(s))
    myList.logEnd("Function " + functionName)
    if verbose:
        print(df.head)
    return df


# Remplacement du manque de valeur par la valeur par défaut (1)
def cleanNaValue(df, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    df = df.copy()
    if verbose: print("\n",df.isna().sum())

    df = df.fillna(1)

    if verbose:
        print("\n",df.isna().sum())
    myList.logEnd("Function " + functionName, verbose)
    return df


# Passage des types en entier
def convertEvalutionToInt(df, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    df = df.copy()
    if verbose: print("\n",df.dtypes)

    for columnName in df.columns:
        if columnName in columnNameNotStudient:
            continue
        else:
            df[columnName].astype(dtype='int64')
    if verbose:
        print(df.dtypes)
    myList.logEnd("Function " + functionName, verbose)
    return df


# Faire la moyenne de la classe (par ligne)
def addMeanColumn(df, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    df = df.copy()

    # Faire la moyenne des élèves par ligne
    df["MEAN"] = df.mean(axis=1, numeric_only=True)

    if verbose:
        print(df[['Code Rubrique', 'Code sous rubrique', 'Unité d’apprentissage ', "MEAN"]])
    myList.logEnd("Function " + functionName, verbose)
    return df


def showRadar(df, columnName, people=None, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName)
    maxValue = 4
    properties = df[columnName].unique()
    valuesList = {}
    valuesList["Expected"] = ([3] * len(properties))
    valuesList["Class mean"] = df.groupby(columnName)['MEAN'].mean()
    title = columnName
    if people != None:
        valuesList[people] = df.groupby(columnName)[people].mean()
        title = columnName + " " + people
    myGraph.showGraphRadar(title, properties, valuesList, maxValue, verbose=False)
    myList.logEnd("Function " + functionName, True)

# Présenter quelques statistiques:
# - Les moyennes par rubrique, sous-rubrique et unité d’apprentissage
# - Par apprenant, la moyenne de ses évaluations
# - La moyenne globale
# - La médiane globale
def exercice2(df, verbose):
    df_group = df.groupby('Code Rubrique')['MEAN'].mean()
    print("Moyenne par Rubrique", df_group)

    # - Les moyennes par sous-rubrique
    df_group = df.groupby("Code sous rubrique")['MEAN'].mean()
    print("Moyenne par Code sous rubrique", df_group)

    # - Les moyennes par Unité d’apprentissage
    print("Unité d’apprentissage ", df[['Code Rubrique', "Code sous rubrique", 'MEAN']])

    # Par apprenant, la moyenne de ses évaluations
    print("La moyenne par personne", df.mean(axis=0, numeric_only=True))
    # - La moyenne globale
    print("La moyenne globale", df.mean(axis=0, numeric_only=True)["MEAN"])

    print("La médiane par personne", df.median(axis=0, numeric_only=True))
    # - La médiane globale
    print("La médiane globale", df.median(axis=0, numeric_only=True).median())



# 1. Charger vos données dans un DataFrame Pandas
df = pd.read_csv(fileName)

# Préparation des données
df = renameIndex(df, verbose=verbose)
df = cleanNaValue(df, verbose=verbose)
# Passage des types en entier
df = convertEvalutionToInt(df, verbose=verbose)

# Définir et alimenter une structure de données qui permet d’exploiter le tableau renseigné
# Précalculer certaines données
df = addMeanColumn(df, verbose=verbose)

if verbose:
    exercice2(df, verbose)

showRadar(df, 'Code Rubrique', people="Aurélie", verbose=verbose)
showRadar(df, 'Code sous rubrique', people="Aurélie", verbose=verbose)

# Sauvegarder le dataframe dans un fichier CSV
df.to_csv(resFileName)

print("END")


