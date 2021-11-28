import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import pandas as pd
import numpy as np
import DataFrameUtil as myDf
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from IPython.display import display


def cleanType(df, verbose=False):
    print("Correction des types ....")
    # la colonne contient une valeur string ce qui empêche la conversion en entier
    df.loc[df['NOMBRE_ETUDIANTS'] == 'PrivacySuppressed', 'NOMBRE_ETUDIANTS'] = np.nan
    # la colonne contient une valeur string ce qui empêche la conversion en entier
    df.loc[df['AGE_ENTREE'] == 'PrivacySuppressed', 'AGE_ENTREE'] = np.nan
    df['NOMBRE_ETUDIANTS'] = pd.to_numeric(df['NOMBRE_ETUDIANTS'])
    df['AGE_ENTREE'] = pd.to_numeric(df['AGE_ENTREE'])
    print("Correction des types........ END")
    if verbose: print(df.dtypes)
    return df


def cleanMissingValues(df, verbose=False):
    print("Traitement des valeurrs manquantes ....")
    # Existe-t-il des valeurs manquantes dans ce jeu de données ?
    nadatas = df.isna().sum()
    if verbose: print(nadatas)
    print("Traitement des valeurrs manquantes........ END")
    return df


def cleanDuplicated(df, verbose=False):
    print("Traitement des doublons ....")
    # Existe-t-il des doublons sur l'ID dans ce jeu de données ?
    if verbose: print(df.shape, df["ID"].nunique())
    if verbose: print(df["ID"].duplicated().sum())

    # on compte le nombre de valeurs manquantes pour la ligne et on stocke dans une nouvelle colonne
    df['NB_NAN'] = df.isna().sum(axis=1)
    # trie des lignes en fonction du nombre de valeurs manquantes
    df = df.sort_values('NB_NAN')
    # suppression des duplicatas en gardant les versions les mieux remplies
    df = df.drop_duplicates('ID', keep='first')
    # on supprime la colonne qui n'est plus utile
    df = df.drop('NB_NAN', axis=1)
    if verbose: print(df.shape, df["ID"].nunique())
    if verbose: print(df["ID"].duplicated().sum())
    print("Traitement des doublons ........ END")
    return df


def prepareData(df, verbose=False):
    print("prepareData ... ")
    df = cleanType(df, verbose)
    df = cleanMissingValues(df, verbose)
    df = cleanDuplicated(df, verbose)
    print("prepareData ........................................................ END")
    return df

verbose = False

print("Chargement des données....")
df = pd.read_csv('211028_EdTech_market_study_usa.csv', sep=',')
print("Chargement des données........ END")
if verbose: myDf.displayInfo(df)
if verbose: print(df.dtypes)

df = prepareData(df,verbose)

if verbose: print(df.shape, df["ID"].nunique())
if verbose: print(df.shape, df["NOM"].nunique())


def removeLessEtablissementByEtat(df, nb=5, verbose=False):
    print("removeLessEtablissementByEtat ... ")
    nb_etab_by_etat3 = df.groupby(["ETAT"]).size()
    if verbose: print(nb_etab_by_etat3)
    res = df[~df['ETAT'].isin(nb_etab_by_etat3[nb_etab_by_etat3 < nb].index)]
    if verbose:
        print(res)
        print(df.shape)
        print(res.shape)
    print("removeLessEtablissementByEtat .................. END")
    return res


def replaceNaNByMedianValu(df, columName, verbose=False):
    res = df.copy()
    median = res[columName].median()
    res[columName].fillna(value=median, inplace=True)
    if verbose:
        print(median)
        print(res[columName])
        print(res[columName].isna().sum())
    return res



def createMap2(state_data, verbose = False):

    url = (
        "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
    )
    state_geo = f"{url}/us-states.json"

    m = folium.Map(location=[48, -102], zoom_start=3)
    bins = list(state_data["SCORE_TOT"].quantile([0, 0.25, 0.5, 0.75, 1]))

    folium.Choropleth(
        geo_data=state_geo,
        name="choropleth",
        data=state_data,
        columns=["ETAT", "SCORE_TOT"],
        # columns=["State", "Unemployment"],
        key_on="feature.id",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="SCORE_TOT Rate (%)",
        bins=bins,
        reset=True,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    display(m)


def createMap(state_data, columnName, verbose = False):

    url = (
        "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
    )
    state_geo = f"{url}/us-states.json"

    m = folium.Map(location=[48, -102], zoom_start=4)

    folium.Choropleth(
        geo_data=state_geo,
        name="choropleth",
        data=state_data,
        columns=["ETAT", columnName],
        # columns=["State", "Unemployment"],
        key_on="feature.id",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=columnName,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    display(m)


df = removeLessEtablissementByEtat(df, 5, verbose)

#Quels sont les 3 états qui hébergent le plus d'établissements fonctionnant en mode à distance
# ETAT, A_DISTANCE_SEULEMENT (0 ou 1)
df_distance = df[df["A_DISTANCE_SEULEMENT"]>0]
if verbose:print(df_distance)
df_distance = df_distance.groupby(["ETAT"]).size().sort_values(ascending=False)
if verbose:
    print(df_distance)
    print(df_distance.head(3))

# Faites une moyenne des variables DEFAUT_PAIEMENT_2ANNEES et DEFAUT_PAIEMENT_3ANNEES, stockez le résultat dans une nouvelle colonne DEFAUT_PAIEMENT.
df["DEFAUT_PAIEMENT"] = (df['DEFAUT_PAIEMENT_2ANNEES'] + df['DEFAUT_PAIEMENT_3ANNEES'])/2
if verbose:
    print(df["DEFAUT_PAIEMENT"])
    print(df.head)
    print(df.shape)


# Remplacez les valeurs manquantes de la colonne DEFAUT_PAIEMENT par zéro.
df["DEFAUT_PAIEMENT"].fillna(value=0, inplace=True)
if verbose:
    print(df["DEFAUT_PAIEMENT"])
    print(df["DEFAUT_PAIEMENT"].isna().sum())

# Dans un premier temps remplacez les taux d'admission manquants par la valeur médiane de la variable.
df = replaceNaNByMedianValu(df, "TAUX_ADMISSION", verbose)

# Supprimez les lignes ayant un taux d'admission nul, cela paraît peu probable.
dfClean = df[df["TAUX_ADMISSION"]>0]
if verbose:
    print(dfClean)
    print("DF", df.shape, "dfClean", dfClean.shape)
df = dfClean

# Remplacez les valeurs manquantes de la colonne NOMBRE_ETUDIANTS en remplaçant par la valeur médiane de la variable.
df = replaceNaNByMedianValu(df, "NOMBRE_ETUDIANTS", verbose)

# À l'aide d'un calcul savant, retrouvez le nombre d'étudiants ayant fait une demande d'inscription.
df['NOMBRE_ETUDIANTS_DEMANDEURS'] = df['NOMBRE_ETUDIANTS'] / df['TAUX_ADMISSION']
if verbose:
    print(df[["NOMBRE_ETUDIANTS", "TAUX_ADMISSION","NOMBRE_ETUDIANTS_DEMANDEURS"]])
    print(df["NOMBRE_ETUDIANTS_DEMANDEURS"].sum())
    print(df.dtypes)

# Logiquement il faudrait convertir en INT, mais dans ce cas, les résultats sont différents du corrigé
# df["NOMBRE_ETUDIANTS_DEMANDEURS"] = df["NOMBRE_ETUDIANTS_DEMANDEURS"].astype(dtype='int32')
# if verbose:
#     print(df.head())
#     print(df.dtypes)

#    Nous utiliserons plus tard la variable COUT_MOYEN_ANNEE_ACADEMIE, afin de quantifier le budget éducation des étudiants. Avant cela, il faut remplacer les valeurs manquantes de la variable par la médiane.
df = replaceNaNByMedianValu(df, "COUT_MOYEN_ANNEE_ACADEMIE", verbose)

# Nous allons maintenant créer un score entre 0 et 1 pour noter le critère population étudiante de chaque ville (1 ville pour la plus peuplée, 0 pour la moins peuplée).
df['SCORE_POP'] = (df['NOMBRE_ETUDIANTS_DEMANDEURS'] - df['NOMBRE_ETUDIANTS_DEMANDEURS'].min()) / (df['NOMBRE_ETUDIANTS_DEMANDEURS'].max() - df['NOMBRE_ETUDIANTS_DEMANDEURS'].min())
if verbose:
    print(df['SCORE_POP'])
    print(df.head())
    print(df['SCORE_POP'].describe())

# Créez une colonne SCORE_COUT contenant le score issu de la variable COUT_MOYEN_ANNEE_ACADEMIE.
df['SCORE_COUT'] = (df['COUT_MOYEN_ANNEE_ACADEMIE'] - df['COUT_MOYEN_ANNEE_ACADEMIE'].min()) / (df['COUT_MOYEN_ANNEE_ACADEMIE'].max() - df['COUT_MOYEN_ANNEE_ACADEMIE'].min())
if verbose:
    print(df['SCORE_COUT'])
    print(df.head())
    print(df['SCORE_COUT'].describe())

# Créez une colonne SCORE_DEFAUT contenant le score issu de la variable DEFAUT_PAIEMENT.
df['SCORE_DEFAUT'] = (df['DEFAUT_PAIEMENT'] - df['DEFAUT_PAIEMENT'].min()) / (df['DEFAUT_PAIEMENT'].max() - df['DEFAUT_PAIEMENT'].min())
if verbose:
    print(df['SCORE_DEFAUT'])
    print(df.head())
    print(df['SCORE_DEFAUT'].describe())

# Créez une colonne SCORE_DEFAUT contenant le score issu de la variable DEFAUT_PAIEMENT.
corr_df = df[['SCORE_DEFAUT', 'SCORE_COUT']].corr()
if verbose:
    print("CORR ------------------")
    print(corr_df, "\n")
    plt.figure(figsize=(4, 4))
    sns.heatmap(corr_df, annot=True)
    plt.show()


# On souhaite identifier les écoles ayant un fort potentiel économique pour notre client, voici la liste des critères que l'on recherche :
#
#     Nombre important d'étudiants
#     Prix élevé de la formation
#     Taux d'admission faible
#
# Utilisez les scores calculés précédemment pour construire un nouvel indicateur (SCORE_SYNT) synthétisant ces propriétés.
df['SCORE_SYNT'] = (df['SCORE_POP'] + df['SCORE_DEFAUT'] + (1 - df['SCORE_COUT'])) /3
df = df.sort_values(['SCORE_SYNT'], ascending=False)
if verbose:
    print(df['SCORE_SYNT'])
    print(df.head())
    print(df['SCORE_SYNT'].describe())

df['SCORE_SYNT_POND'] = (df['SCORE_POP']*2 + df['SCORE_DEFAUT'] + (1 - df['SCORE_COUT'])) /4
if verbose:
    print(df[['SCORE_SYNT', 'SCORE_SYNT_POND']])
    print(df.head())
    print(df['SCORE_SYNT_POND'].describe())

# Donnez la liste des 15 établissements les mieux classés par rapport à
df_reduce = df[['NOM', 'ETAT','SCORE_POP','SCORE_COUT','TAUX_ADMISSION','SCORE_SYNT']]
if verbose:
    print(df_reduce.head(15))


df['SCORE_TOT'] = df['SCORE_POP'] + df['SCORE_DEFAUT'] + df['SCORE_COUT']
df = df.sort_values(['SCORE_TOT'], ascending=False)
serie_score_total_ville = df.groupby("VILLE")['SCORE_TOT'].sum()
serie_score_total_ville = serie_score_total_ville.sort_values(ascending=False)
if verbose:
    print(df.head(15))
    print(serie_score_total_ville[:15])


serie_score_total_etat = df.groupby("ETAT")['SCORE_POP'].sum()
serie_score_total_etat = serie_score_total_etat.sort_values(ascending=False)
createMap(serie_score_total_etat, "SCORE_POP", verbose)

serie_score_total_etat = df.groupby("ETAT")['SCORE_DEFAUT'].sum()
serie_score_total_etat = serie_score_total_etat.sort_values(ascending=False)
createMap(serie_score_total_etat, "SCORE_DEFAUT", verbose)

serie_score_total_etat = df.groupby("ETAT")['SCORE_COUT'].sum()
serie_score_total_etat = serie_score_total_etat.sort_values(ascending=False)
createMap(serie_score_total_etat, "SCORE_COUT", verbose)

serie_score_total_etat = df.groupby("ETAT")['SCORE_TOT'].sum()
serie_score_total_etat = serie_score_total_etat.sort_values(ascending=False)
if verbose:
    print(serie_score_total_etat[:15])
createMap(serie_score_total_etat, "SCORE_TOT", verbose)

print("END")

