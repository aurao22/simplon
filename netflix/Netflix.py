"""
Contexte du projet

    Lecture du fichier :
    Dimension du dataframe
        Combien y a-t-il de films dans ce jeu de données ?
        Combien y a-t-il de séries ?
        Générer le résumé statistique du dataframe
        Compter les valeurs manquantes

    Exploration des valeurs manquantes :
        Sur la colonne des directeurs de production
        Sur la colonne des acteurs
        Suppression des lignes dupliquées (s'ils existents)

    Questions :
    Combien de films/séries ont été produit par les Etats-Unis ? Par la France ?
    Quel est le film avec la durée la plus longue sur Netflix ?
    Etude des catégories avec le plus de contenu
    Quels directeurs ont produit le plus de films/séries disponibles sur Netflix ?
    Est-ce que Jan Suter travaille souvent avec les mêmes acteurs ?

    Représenter les 10 pays qui ont produits le plus de contenus disponibles sur Netflix, avec le nombre de contenus par pays
    Tracer un graphe à barres du nombre de films/séries par classement de contenu (rating)
    Tracer un graphique qui montre l'évolution du nombre de films/séries produits par les 10 pays les plus producteurs de contenus sur Netflix, au fil des ans
    Distribution de la durée des films disponibles sur Netflix
    Tracer un graphique représentant le nombre de séries par modalité de nombre de saisons

Modalités pédagogiques
    Pour réaliser ce projet, vous allez utiliser JupyterLab avec des librairies data science en python telles que Pandas, Matplotlib et SeaBorn.
    Livrables
    Un notebook comportant les analyses pré-exploratoires réalisées (non cleané, pour comprendre votre démarche).
    Interpreter vos résultats dans le notebook.
"""
from builtins import set
from time import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from geopy.geocoders import Nominatim
# import folium
# from folium.plugins import MarkerCluster
# from IPython.display import display

# ---------------------------------------------------------------------------------------------
#                               Variables Globales
# ---------------------------------------------------------------------------------------------
colorsPaletteSeaborn = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]

lat_long = {('IT','EU'):(41.871940, 12.567380),         # Italie
            ('JP','AS'): (34.886306, 134.379711),       # ('JP', 'AS') Japan nan
            ('CZ','EU'): (49.817492, 15.472962), 	    # ('CZ', 'EU') Czech Republic nan
            ('VE','SA'): (6.423750, -66.589730),		# ('VE', 'SA') Venezuela nan
            ('NP','AS'): (28.394857, 84.124008),		# ('NP', 'AS') Nepal nan
            ('SY','AS'): (34.802075, 38.996815), 		# ('SY', 'AS') Syria nan
            ('IE','EU'): (53.412910, -8.243890),  		# ('IE', 'EU') Ireland nan
            ('UY','SA'): (-32.522779, -55.765835),   	# ('UY', 'SA') Uruguay nan
            ('KY','NA'): (19.313300, -81.254600), 		# ('KY', 'NA') Cayman Islands nan
            ('JO','AS'): (30.585164, 36.238414), 		# ('JO', 'AS') Jordan nan
            ('ZW','AF'): (-19.015438, 29.154857), 		# ('ZW', 'AF') Zimbabwe nan
            ('FI','EU'): (61.924110, 25.748151),		# ('FI', 'EU') Finland nan
            ('MW','AF'): (-13.254308, 34.301525),  		# ('MW', 'AF') Malawi nan
            ('PY','SA'): (-23.442503, -58.443832), 		# ('PY', 'SA') Paraguay nan
            ('UA','EU'): (44.874119, 33.151245), 		# ('UA', 'EU') Ukraine nan
            ('EC','SA'): (-1.831239, -78.183406),		# ('EC', 'SA') Ecuador nan
            ('AM','AS'): (40.069099, 45.038189), 		# ('AM', 'AS') Armenia nan
            ('LK','AS'): (7.873592, 80.773137),			# ('LK', 'AS') Sri Lanka nan
            ('PR','NA'): (18.220833, -66.590149),		# Puerto Rico
            ('GB','EU'): (52.3555177, -1.1743197),      # United Kingdom
            ('UG','AF'): (1.373333, 32.290275)			# ('UG', 'AF') Uganda nan
            }

verbose = False

# ---------------------------------------------------------------------------------------------
#                               Chargement du fichier
# ---------------------------------------------------------------------------------------------
# Lecture du fichier :
print("Chargement des données....")
df = pd.read_csv('netflix_titles.csv', sep=',')
print("Chargement des données........ END")


# ---------------------------------------------------------------------------------------------
#                               PREPARATION DES DONNEES
# ---------------------------------------------------------------------------------------------
#                                     Dimensions
def get_nb_of_type(df, type, verbose=False):
    """
    Retourne le nombre d'éléments du type demandé en paramètre
    :param df: DataFrame
    :param type : String - type recherché
    :param verbose : Boolean - True pour mode debug
    :return: le nombre d'éléments du type demandé en paramètre
    """
    t0 = time()
    nb_type = 0
    types_count = df["type"].value_counts()
    for t,v in types_count.items():
        if t == type:
            nb_type = v
            break
    t1 = time() - t0
    print("get_nb_of_type in {0:.3f} secondes................... END".format(t1))
    return nb_type


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("|             Dimension du dataframe            |")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Dimension du dataframe", df.shape)
print("Combien par type:\n", df["type"].value_counts())
print("Combien y a-t-il de films dans ce jeu de données ?", get_nb_of_type(df, "Movie", verbose))
print("Combien y a-t-il de séries ?", get_nb_of_type(df, "TV Show", verbose))
print("Générer le résumé statistique du dataframe")
print(df.describe())


# ---------------------------------------------------------------------------------------------
#                               PREPARATION DES DONNEES
# ---------------------------------------------------------------------------------------------
#                               Corrections des types
def cleanType(df, verbose=False):
    """
    Nettoie et Transforme rating en catégorie
    :param df: DataFrame
    :param verbose : Boolean - True pour mode debug
    :return: DataFrame - a new clean DataFrame
    """
    t0 = time()
    print("cleanType ....")
    df = df.copy()

    # Traitement du type
    df["type"] = df["type"].astype('category')

    # Traitement du rating
    if verbose: print("rating : ", df["rating"].unique())
    # Il y a un décalage pour certaines notes, donc il faut les corriger avant de changer le type
    for rating in df["rating"].unique():
        if " min" in str(rating):
            # Il faut corriger
            df.loc[df["rating"] == rating, 'duration'] = rating
            df.loc[df["rating"] == rating, 'rating'] = np.nan
    if verbose: print("rating : ", df["rating"].unique())
    # Conversion des notes en catégorie
    df["rating"] = df["rating"].astype('category')

    # Traitement de la date
    df["date_added"] = pd.to_datetime(df['date_added'])

    # Traitement de la duration
    df["duration time"] = df["duration"].str.replace(' min','')
    df["duration time"] = df["duration time"].str.replace(' Seasons', '')
    df["duration time"] = df["duration time"].str.replace(' Season', '')
    if verbose:
        print(df[["duration","duration time"]])
        print(df.dtypes)
    df["duration time"] = pd.to_numeric(df["duration time"])

    # Traitement des pays
    df["country"] = df["country"].str.replace('West Germany', 'Germany')
    df["country"] = df["country"].str.replace('East Germany', 'Germany')
    df["country"] = df["country"].str.replace('Soviet Union', 'Russian Federation')
    df["country"] = df["country"].str.replace('Vatican City', 'Holy See')

    # Traitement de l'année
    print(df["release_year"].unique())
    for year in df["release_year"].unique():
        df.loc[df["release_year"] == year, "annee"] = datetime(year=year, month=1, day=1)

    if verbose: print(df.dtypes)
    t1 = time() - t0
    print("cleanType in {0:.3f} secondes................... END".format(t1))
    return df


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("|             Corrections des types            |")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Correction des types
print(df.dtypes)
df = cleanType(df, verbose)
print(df.dtypes)
print(df.describe())


# ---------------------------------------------------------------------------------------------
#                               PREPARATION DES DONNEES
# ---------------------------------------------------------------------------------------------
#                           Exploration des valeurs manquantes
#                           Traitement des données dupliquées
def cleanDuplicated(df, verbose=False):
    """
    Delete duplicate datas
    :param df: DataFrame
    :param verbose : Boolean - True pour mode debug
    :return: DataFrame - a new clean DataFrame
    """
    t0 = time()
    print("cleanDuplicated ....")
    df = df.copy()
    # Existe-t-il des doublons dans ce jeu de données ?
    if verbose:
        print(df.nunique())
        print(df.duplicated())
        print("df.duplicated().sum() =", df.duplicated().sum())
        print("show_id")
        print(df.shape, df["show_id"].nunique())
        print(df["show_id"].duplicated().sum())
    t1 = time() - t0
    print("cleanDuplicated in {0:.3f} secondes................... END".format(t1))
    return df


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("|     Exploration des valeurs manquantes        |")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Compter les valeurs manquantes")
print(df.isna().sum())
print("Sur la colonne des directeurs de production", df["director"].isna().sum())
print("Sur la colonne des acteurs", df["cast"].isna().sum())
print("Suppression des lignes dupliquées (s'ils existents)")
df = cleanDuplicated(df, verbose)
print(df.isna().sum())


def process_country(df, country, verbose=False):
    """
    Créé une copie du DataFrame, y ajoute une colonne pour le pays souhaité avec un boolean si le contenu vient de ce pays = True
    :param df: DataFrame
    :param verbose : Boolean - True pour mode debug
    :return: DataFrame - a new DataFrame avec la colonne pour le pays
    """
    t0 = time()
    df_country = df.copy()
    df_country[country] = df_country["country"].str.contains(country, case=False)
    if verbose:
        nb_total = df_country[country].value_counts()[True]
        select = df_country[df_country["type"] == "Movie"].index.intersection(df_country[df_country[country] == True].index)
        nb_films = len(select)
        select = df_country[df_country["type"] == "TV Show"].index.intersection(df_country[df_country[country] == True].index)
        nb_series = len(select)
        print(country, "nb_total :", nb_total, " - nb_films :", nb_films, " - nb_series :", nb_series)
    t1 = time() - t0
    print("process_country in {0:.3f} secondes................... END".format(t1))
    return df_country


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("|                      Questions                |")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Combien de films/séries ont été produit par les Etats-Unis ? Par la France ?")
if verbose: print("country : ", df["country"].unique())

# Identification des pays via ajout de colonne
states = 'United States'
france = 'France'

df_states = process_country(df, states, True)
df_france = process_country(df_states, france, True)

# Films en collaboration France / Etats Unis
select = df_france[df_france[states] == True].index.intersection(df_france[df_france[france] == True].index)
print("Films en colaboration France / Etats Unis :", len(select.value_counts()))


# ---------------------------------------------------------------------------------------------
#                               QUESTIONS - VERSION 2
# ---------------------------------------------------------------------------------------------
def process_country_version2(df, country, verbose=False):
    """
    Créé une copie du DataFrame, y ajoute une colonne pour le pays souhaité avec un boolean si le contenu vient de ce pays = True
    :param df: DataFrame
    :param country: String - Country name
    :param verbose : Boolean - True pour mode debug
    :return: DataFrame - a new DataFrame avec la colonne pour le pays
    """
    t0 = time()
    df_copy = df.copy()
    df_country = df_copy["country"].str.split(",", n=0, expand=True)
    if verbose:
        print(df_country.shape)
        print(df_country.columns)
    nb_col = df_country.shape[1]
    df_country[country] = df_country[0].str.strip() == country
    if verbose:
        print(df_country[country])
        print(df_country.columns)
        print(df_country[country].unique())

    for i in range(1, nb_col):
        df_country[i] = df_country[i].str.strip() == country
        df_country[country] = df_country[country] | df_country[i]

    pays = df_country[country]
    df_copy[country] = pays

    if verbose:
        nb_total = df_copy[country].value_counts()[True]
        select = df_copy[df_copy["type"] == "Movie"].index.intersection(df_copy[df_copy[country] == True].index)
        nb_films = len(select)
        select = df_copy[df_copy["type"] == "TV Show"].index.intersection(df_copy[df_copy[country] == True].index)
        nb_series = len(select)
        print(country, "nb_total :", nb_total, " - nb_films :", nb_films, " - nb_series :", nb_series)
    t1 = time() - t0
    print("process_country in {0:.3f} secondes................... END".format(t1))
    return df_copy


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("|              Questions - VERSION 2            |")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Combien de films/séries ont été produit par les Etats-Unis ? Par la France ?")

df_states = process_country_version2(df, states, True)
df_france = process_country_version2(df_states, france, True)

# Films en colaboration France / Etats Unis
select = df_france[df_france[states] == True].index.intersection(df_france[df_france[france] == True].index)
print("Films en colaboration France / Etats Unis :", len(select.value_counts()))


print("Quel est le film avec la durée la plus longue sur Netflix ?")
select = df[df["type"] == "Movie"]
duree = select["duration time"].max()
print("le film avec la durée la plus longue (", duree, ")")
print(select[select["duration time"] == duree])


def count_unique_data(df, column_name, cat, verbose=False):
    """
    Compte le nombre de contenu unique pour la donnée reçue dans la colonne reçue
    :param df: DataFrame
    :param column_name : String : nom de la colonne à traiter
    :param cat : String - Valeur à compter
    :param verbose : Boolean - True pour mode debug
    :return: Int - nombre de contenu
    """
    t0 = time()
    df = df.copy()
    serie = df[column_name].str.contains(cat, case=False)
    nb_total = serie.value_counts()[True]
    if verbose:
        t1 = time() - t0
        print("count_unique_data in {0:.3f} secondes................... END".format(t1))
    return nb_total


def get_unique_data(df, column_name, verbose=False):
    """
    Compte le nombre de contenu unique pour la colonne reçue et revoi le
    :param df: DataFrame
    :param column_name : String : nom de la colonne à traiter
    :param verbose : Boolean - True pour mode debug
    :return: dic{<data_value>:<data_count>}, dic{<data_count>:[<data_values>]}, max int
    """
    t0 = time()
    df = df.copy()
    categories_list = {}
    categories_by_size = {}
    max = 0
    if verbose: print(column_name, " : ", df[column_name].unique())
    for cats_str in df[column_name].unique():
        if cats_str is None or cats_str == np.nan or type(cats_str) != str:
            continue
        cats = cats_str.split(",")
        for cat in cats:
            cat = cat.strip()
            if len(cat) > 0 and cat not in categories_list:
                # comptage du nombre d'élément par catégorie
                nb = count_unique_data(df, column_name, cat, False)
                categories_list[cat] = nb
                if not categories_by_size.get(nb, False):
                    categories_by_size[nb] = []
                categories_by_size[nb].append(cat)
                if verbose:
                    print(nb," : " , cat)
                if nb > max:
                    max = nb
    t1 = time() - t0
    print("get_unique_data in {0:.3f} secondes................... END".format(t1))
    return categories_list, categories_by_size, max


def get_unique_data_version(df, column_name, verbose=False):
    """
    Compte le nombre de contenu unique pour la colonne reçue et revoi le
    :param df: DataFrame
    :param column_name : String : nom de la colonne à traiter
    :param verbose : Boolean - True pour mode debug
    :return: dic{<data_value>:<data_count>}, dic{<data_count>:[<data_values>]}, max int
    """
    t0 = time()
    df = df.copy()
    df_data = df[column_name].str.split(",", n=0, expand=True)
    # df_data.to_csv("netflix_categories_dataframe.csv")
    set_liste_data = set()
    if verbose:
        print(df_data.shape)
        print(df_data.columns)
    nb_col = df_data.shape[1]
    # Construction de la liste de données uniques
    for i in range(nb_col):
        set_liste_data = set_liste_data | set(df_data[i].str.strip())

    if verbose:
        print(len(set_liste_data))
        print(set_liste_data)

    data_list = {}
    data_by_size = {}
    max = 0

    for columnName in df_data.columns:
        grp = df_data.groupby(columnName)[columnName].value_counts()
        if verbose:
            print(grp)
        for k,value in grp.items():
            cat = k[1].strip()
            nb = data_list.get(cat, 0) + value
            data_list[cat] = nb

    for cat,nb in data_list.items():
        if not data_by_size.get(nb, False):
            data_by_size[nb] = []
        data_by_size[nb].append(cat)
        if verbose:
            print(nb," : " , cat)
        if nb > max:
            max = nb

    t1 = time() - t0
    print("get_unique_data_version in {0:.3f} secondes................... END".format(t1))
    return data_list, data_by_size, max


def print_dic_by_size(dico, limit=5, verbose=False):
    """
    Affiche le dictionnaire par ordre décroissant jusqu'à la limite ou la taille du dico
    :param dico: {dictionnaire(key=int, value} : dictionnaire à afficher
    :param limit: int = nombre d'éléments à afficher (si verbose = True, affiche tout le dictionnaire)
    :param verbose : Boolean - True pour mode debug
    :return: None
    """
    keys = sorted(dico.keys(), reverse=True)
    i = limit
    if verbose:
        i = len(keys)
    for k in keys:
        print(k, " pour ", dico[k])
        i -= 1
        if i == 0:
            break


def categories(df, verbose=False, version=1):
    """
    Etude des catégories avec le plus de contenu.
    :param df: DataFrame
    :param verbose : Boolean - True pour mode debug
    :return: None
    """
    t0 = time()
    df = df.copy()
    print("Etude des catégories avec le plus de contenu.")
    # Construction de la liste des catégories
    if(version == 2):
        categories_list, categories_by_size, max2 = get_unique_data_version(df, "listed_in", verbose)
    else:
        categories_list, categories_by_size, max2 = get_unique_data(df, "listed_in", verbose)
    t1 = time() - t0
    print("Nb Catégories :", len(categories_list), 'in {0:.3f} secondes'.format(t1))
    print("Catégories avec le plus de contenus :", max2)
    print("MAX=", max2, "soit :", categories_by_size[max2])
    print_dic_by_size(categories_by_size, 10, verbose)
    t1 = time() - t0
    print("categories in {0:.3f} secondes................... END".format(t1))


def directors(df, verbose=False, version=1):
    """
    Etude des directeurs ont produit le plus de films/séries avec le plus de contenu.
    :param df: DataFrame
    :param verbose : Boolean - True pour mode debug
    :return: None
    """
    t0 = time()
    df = df.copy()
    print("Quels directeurs ont produit le plus de films/séries disponibles sur Netflix ?")
    # Construction de la liste des catégories
    if (version == 2):
        directors_list, directors_by_size, max_dir = get_unique_data_version(df, "director", verbose)
    else:
        directors_list, directors_by_size, max_dir = get_unique_data(df, "director", verbose)
    t1 = time() - t0
    print("Nb directors :", len(directors_list), 'in {0:.3f} secondes'.format(t1))
    print("Directors avec le plus de contenus :", directors_by_size[max_dir], "avec :", max_dir, "contenus")
    keys = sorted(directors_by_size.keys(), reverse=True)

    # Pour éviter de tout afficher
    print_dic_by_size(directors_by_size, 10, verbose)
    t1 = time() - t0
    print("directors in {0:.3f} secondes................... END".format(t1))


def james_suter(df, verbose=False, version=1):
    """
    Etude sur Jan Suter travaille souvent avec les mêmes acteurs.
    :param df: DataFrame
    :param verbose : Boolean - True pour mode debug
    :return: None
    """
    t0 = time()
    print("Est-ce que Jan Suter travaille souvent avec les mêmes acteurs ?")
    director_name = "Jan Suter"
    df = df.copy()
    if verbose:
        print("james_suter - BEFORE drop NA", df.shape)
    df = df.dropna(subset=["director"])
    if verbose:
        print("james_suter - AFTER drop NA", df.shape)
    select = df[df["director"].str.contains(director_name, case=False)]
    print("james_suter - nb contenus", select.shape)
    # Construction de la liste des catégories
    if (version == 2):
        actors_list, actors_by_size, max_dir = get_unique_data_version(select, "cast", verbose)
    else:
        actors_list, actors_by_size, max_dir = get_unique_data(select, "cast", verbose)
    t1 = time() - t0
    print("Nb actors :", len(actors_list), 'in {0:.3f} secondes'.format(t1))
    print("Directors avec le plus de contenus :", actors_by_size[max_dir], "avec :", max_dir, "contenus")
    keys = sorted(actors_by_size.keys(), reverse=True)

    # Pour éviter de tout afficher
    print_dic_by_size(actors_by_size, 10, verbose)

    t1 = time() - t0
    print("james_suter in {0:.3f} secondes................... END".format(t1))


#categories(df, verbose, version=1)
#directors(df, verbose, version=1)
#james_suter(df, True, version=1)

categories(df, verbose, version=2)
directors(df, verbose, version=2)
james_suter(df, verbose, version=2)


def getAColorsPaletteSeaborn():
    c = sample(colorsPaletteSeaborn, 1)
    return c[0]


# 1. Conversion to Alpha 2 codes and Continents
def get_continent(col):
    try:
        cn_a2_code =  country_name_to_country_alpha2(col)
    except:
        cn_a2_code = 'Unknown'
        if col == 'United Kingdom':
            cn_a2_code = 'GB'
    try:
        cn_continent = country_alpha2_to_continent_code(cn_a2_code)
    except:
        cn_continent = 'Unknown'
        if col == 'United Kingdom':
            cn_continent = 'EU'
    return (cn_a2_code, cn_continent)


# 2. Get longitude and latitude
def geolocate(country, geolocator):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return lat_long.get(country, np.nan)


def displayBarGraphSeabornOneSerie(df, xName, title, verbose=False, show=True, isY=False):
    colors = sns.color_palette(getAColorsPaletteSeaborn())
    if not isY:
        g = sns.barplot(x=df[xName].index,y=df[xName], palette=colors)
        g.set_ylabel("NB " + xName)
        g.set_xlabel(xName + " Index")
    else:
        g = sns.barplot(x=df[xName], y=df[xName].index, palette=colors)
        g.set_xlabel("NB " + xName)
        g.set_ylabel(xName + " Index")
    plt.title(title)
    if show : plt.show(block=False)


# ---------------------------------------------------------------------------------------------
#                               Préparation des données
# ---------------------------------------------------------------------------------------------
def countries_prepare_data(df, verbose=False, version=1):
    """
    Ajoute les données manquantes et corrige les types
    :param df: DataFrame
    :param verbose : Boolean - True pour mode debug
    :return: DataFrame
    """
    t0 = time()
    dfc = df.copy()
    print("Représenter les 10 pays qui ont produits le plus de contenus disponibles sur Netflix, avec le nombre de contenus par pays?")
    # Construction de la liste des catégories
    if version == 2:
        countries_list, countries_by_size, max_dir = get_unique_data_version(dfc.copy(), "country", verbose)
    else:
        countries_list, countries_by_size, max_dir = get_unique_data(dfc.copy(), "country", verbose)
    t1 = time() - t0
    if verbose:
        print("Nb country :", len(countries_list), 'in {0:.3f} secondes'.format(t1))
        print("Countries avec le plus de contenus :", countries_by_size[max_dir], "avec :", max_dir, "contenus")
    keys = sorted(countries_by_size.keys(), reverse=True)

    # Pour éviter de tout afficher
    if verbose:
        i = 10
        for k in keys:
            print(k, " pour ", countries_by_size[k])
            i -= 1
            if i == 0:
                break
    t1 = time() - t0
    if verbose:
        print("countries in {0:.3f} secondes................... END".format(t1))

    # Conversion de notre liste en DataFrame
    countries_df = pd.DataFrame.from_dict(countries_list, orient="index", columns=["nb_contenus"])
    if verbose: print(countries_df)

    # Il faut ajouter codes, country, continent,
    countries_df["Country"] = ""
    countries_df["Continent"] = ""
    countries_df["Latitude"] = ""
    countries_df["Longitude"] = ""
    geolocator = Nominatim(user_agent="catuserbot")
    for c, row in countries_df.iterrows():
        res = get_continent(c)
        if res[0] != 'Unknown' and res[1] != 'Unknown':
            countries_df.loc[c, 'Country'] = res[0]
            countries_df.loc[c, 'Continent'] = res[1]

            geoloc = geolocate(res, geolocator)
            if geoloc != np.nan:
                try:
                    countries_df.loc[c, "Latitude"] = geoloc[0]
                    countries_df.loc[c, "Longitude"] = geoloc[1]
                except TypeError:
                    print("TypeError for :", res, c, geoloc)
            else:
                print("Country not found geoloc :", res, c)
        elif res[0] != 'Unknown':
            countries_df.loc[c, 'Country'] = res[0]
        elif res[1] != 'Unknown':
            countries_df.loc[c, 'Continent'] = res[1]
        else:
            if c == 'Holy See':
                countries_df.loc[c, "Latitude"] = 41.902916
                countries_df.loc[c, "Longitude"] = 12.453389
            else:
                print("Country not known :", c)

    if verbose:
        print(countries_df)
        # countries_df.to_csv("countries.csv")

    # Traitement des valeurs manquantes et corrections des types
    if verbose:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("|     Exploration des valeurs manquantes / correction des types        |")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Compter les valeurs manquantes")
        print(countries_df.shape)
        print(countries_df.isna().sum())
    countries_df = countries_df.dropna(axis=0)
    if verbose:
        print("------")
        print(countries_df.shape)
        print(countries_df.isna().sum())
        print(countries_df.dtypes)
    countries_df["Longitude"] = pd.to_numeric(countries_df["Longitude"])
    countries_df["Latitude"] = pd.to_numeric(countries_df["Latitude"])
    if verbose:
        print(countries_df.dtypes)

    t1 = time() - t0
    print("countries in {0:.3f} secondes................................................... END".format(t1))
    return countries_df, countries_list, countries_by_size


def countries(countries_df, countries_list, countries_by_size, verbose=False):
    """
    8. Représenter les 10 pays qui ont produits le plus de contenus disponibles sur Netflix, avec le nombre de contenus par pays
    :param countries_df: DataFrame
    :param countries_list: Dict {country_name (str) : nb_content (int)}
    :param countries_by_size: Dict {nb_content (int) : country_name (str)}
    :param verbose : Boolean - True pour mode debug
    :return: DataFrame - countries_light
    """
    print("Nb country :", len(countries_list))
    print("8. Représenter les 10 pays qui ont produits le plus de contenus disponibles sur Netflix,"
          " avec le nombre de contenus par pays?")
    keys = sorted(countries_by_size.keys(), reverse=True)

    # Pour éviter de tout afficher
    i = 10
    for k in keys:
        print(k, " pour ", countries_by_size[k])
        i -= 1
        if i == 0:
            break

    ##################### MAP
    # world_map = folium.Map(tiles="cartodbpositron")
    # marker_cluster = MarkerCluster().add_to(world_map)
    # # for each coordinate, create circlemarker of user percent
    # for i in range(len(countries_df)):
    #     lat = countries_df.iloc[i]['Latitude']
    #     long = countries_df.iloc[i]['Longitude']
    #     if lat != "" and long != "":
    #         radius = countries_df.iloc[i]['nb_contenus']
    #         popup_text = str(countries_df.iloc[i]['nb_contenus'])
    #         folium.CircleMarker(location=(lat, long), radius=radius, popup=popup_text, fill=True).add_to(marker_cluster)
    #     else:
    #         print("", end="")
    # # show the map
    # display(world_map)

    # Récupération des 10 pays
    countries_light = countries_df.sort_values("nb_contenus", ascending=False).head(10)

    # define Seaborn color palette to use
    colors = sns.color_palette(getAColorsPaletteSeaborn())

    # create pie chart
    mp.pyplot.pie("nb_contenus", data=countries_light, labels="Country", colors=colors, autopct='%.0f%%')
    mp.pyplot.title("NETFLIX - Countries repartition")
    mp.pyplot.show(block=False)

    # Affichage des graphiques en barre horizontale
    countries_light = countries_df.sort_values("nb_contenus", ascending=False).head(10)
    displayBarGraphSeabornOneSerie(countries_light, 'nb_contenus', "Répartition des contenues par pays",
                                   verbose=verbose, isY=True)
    print("countries ................................................. END")
    return countries_light


countries_df, countries_list, countries_by_size = countries_prepare_data(df, verbose, 2)
countries_light = countries(countries_df, countries_list, countries_by_size, verbose)


def ratings(df, verbose=False):
    print("9. Tracer un graphe à barres du nombre de films/séries par classement de contenu (rating)")
    rating_df = df.copy()
    if verbose:
        print(rating_df.shape)
        print(rating_df.isna().sum())
    rating_df = rating_df.dropna(subset=['rating'], axis="index")
    if verbose:
        print(rating_df.shape)
        print(rating_df.isna().sum())

    sns.countplot(data=rating_df, hue="type", x="rating",
                  order=rating_df['rating'].value_counts(ascending=True).index)
    plt.title("Netflix - Nombre de films/séries par classement de contenu (rating)")
    plt.show(block=False)


ratings(df, verbose)
print(df["rating"].nunique())

# ---------------------------------------------------------------------------------------------
#                               Préparation des données
# ---------------------------------------------------------------------------------------------
def countries_progress_prepare_data(df, countries_light, verbose=False):
    t0 = time()
    country_prog_df = df.copy()
    countries_light = countries_light.copy()

    # Ajout des années en colonnes du DF countries_light
    # les valeurs étant le nombre de films
    years = country_prog_df["release_year"].unique()
    years = years.tolist()
    # Trie des années par ordre chronologique pour le tableau
    years = sorted(years)
    print(years)
    for year in years:
        countries_light[year] = 0

    # Ajoute une colonne pour le pays concerné avec True or False si le pays a contribué
    # Donc 10 colonnes sont ajoutées au dataset
    for pays_name in countries_light.index:
        country_prog_df = process_country_version2(country_prog_df, pays_name, verbose)
        country_prog_df.loc[country_prog_df[pays_name] == False, pays_name] = np.nan
        if verbose:
            print(country_prog_df.shape)
        # Récupération du nombre par an
        group_year = country_prog_df.groupby("release_year")[pays_name].value_counts()
        # Ajout le total dans le countries_light de chaque année
        for y in group_year.index:
            countries_light.loc[pays_name, y] = group_year[y]

    if verbose:
        print(countries_light.shape)
        print(countries_light.columns)
        print(countries_light.head(10))

    # Inversion des colonnes et des lignes
    countries_light_transpose = countries_light.transpose()
    countries_light_transpose.head(10)

    # suppression des colonnes inutiles
    countries_light_transpose_light = countries_light_transpose.drop(
        ['nb_contenus', 'Country', 'Continent', 'Latitude', 'Longitude', 1.0])

    countries_light_transpose_light["idx_annee"] = ""
    for annee in countries_light_transpose_light.index:
        countries_light_transpose_light.loc[annee]["idx_annee"] = "01/01/" + str(annee)

    if verbose:
        print(countries_light_transpose_light.head(10))
        print(countries_light_transpose_light.dtypes)

    # correction des types
    for colname in countries_light_transpose_light.columns:
        if colname != "idx_annee":
            countries_light_transpose_light[colname] = pd.to_numeric(countries_light_transpose_light[colname])
    countries_light_transpose_light["idx_annee"] = pd.to_datetime(countries_light_transpose_light["idx_annee"],
                                                                  dayfirst=True)
    if verbose:
        print(countries_light_transpose_light.dtypes)

    t1 = time() - t0
    print(
        "countries_progress_prepare_data in {0:.3f} secondes................................................... END".format(
            t1))
    return countries_light, years, countries_light_transpose_light


def countries_progress(df, verbose=False):
    print("10. Tracer un graphique qui montre l'évolution du nombre de films/séries produits par les 10 pays "
          "les plus producteurs de contenus sur Netflix, au fil des ans")
    country_prog_df = df.copy()


    plt.title("Netflix - Evolution des contenus par les 10 pays les plus producteurs de contenu")
    plt.show(block=False)


verbose = True
# 10. Tracer un graphique qui montre l'évolution du nombre de films/séries produits par les 10 pays les plus producteurs de contenus sur Netflix, au fil des ans
countries_light, years, countries_light_transpose_light = countries_progress_prepare_data(df, countries_light, verbose)
print(countries_light.head(10))
countries_progress(countries_light, verbose)

print("END")
