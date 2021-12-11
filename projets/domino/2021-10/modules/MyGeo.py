from time import time
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from geopy.geocoders import Nominatim


LATITUDE_LONGITUDE = {('IT', 'EU'):(41.871940, 12.567380),  # Italie
                      ('JP','AS'): (34.886306, 134.379711),  # ('JP', 'AS') Japan nan
                      ('CZ','EU'): (49.817492, 15.472962),  # ('CZ', 'EU') Czech Republic nan
                      ('VE','SA'): (6.423750, -66.589730),  # ('VE', 'SA') Venezuela nan
                      ('NP','AS'): (28.394857, 84.124008),  # ('NP', 'AS') Nepal nan
                      ('SY','AS'): (34.802075, 38.996815),  # ('SY', 'AS') Syria nan
                      ('IE','EU'): (53.412910, -8.243890),  # ('IE', 'EU') Ireland nan
                      ('UY','SA'): (-32.522779, -55.765835),  # ('UY', 'SA') Uruguay nan
                      ('KY','NA'): (19.313300, -81.254600),  # ('KY', 'NA') Cayman Islands nan
                      ('JO','AS'): (30.585164, 36.238414),  # ('JO', 'AS') Jordan nan
                      ('ZW','AF'): (-19.015438, 29.154857),  # ('ZW', 'AF') Zimbabwe nan
                      ('FI','EU'): (61.924110, 25.748151),  # ('FI', 'EU') Finland nan
                      ('MW','AF'): (-13.254308, 34.301525),  # ('MW', 'AF') Malawi nan
                      ('PY','SA'): (-23.442503, -58.443832),  # ('PY', 'SA') Paraguay nan
                      ('UA','EU'): (44.874119, 33.151245),  # ('UA', 'EU') Ukraine nan
                      ('EC','SA'): (-1.831239, -78.183406),  # ('EC', 'SA') Ecuador nan
                      ('AM','AS'): (40.069099, 45.038189),  # ('AM', 'AS') Armenia nan
                      ('LK','AS'): (7.873592, 80.773137),  # ('LK', 'AS') Sri Lanka nan
                      ('PR','NA'): (18.220833, -66.590149),  # Puerto Rico
                      ('GB','EU'): (52.3555177, -1.1743197),  # United Kingdom
                      ('UG','AF'): (1.373333, 32.290275)  # ('UG', 'AF') Uganda nan
                      }


def clean_country_name(df, country_column_name, verbose=False):
    """
    Corrige certains noms de pays
    :param df: DataFrame
    :param country_column_name : String = Nom de la colonne contenant le nom du pays
    :param verbose : Boolean - True pour mode debug
    :return: DataFrame - a new clean DataFrame
    """
    t0 = time()
    print("clean_country_name ....")
    df = df.copy()

    # Traitement des pays
    df[country_column_name] = df[country_column_name].str.replace('West Germany', 'Germany')
    df[country_column_name] = df[country_column_name].str.replace('East Germany', 'Germany')
    df[country_column_name] = df[country_column_name].str.replace('Soviet Union', 'Russian Federation')
    df[country_column_name] = df[country_column_name].str.replace('Vatican City', 'Holy See')
    # df[country_column_name] = df[country_column_name].str.replace('United Kingdom', 'United Kingdom of Great Britain and Northern Ireland')

    if verbose: print(df[country_column_name])
    t1 = time() - t0
    print("clean_country_name in {0:.3f} secondes................... END".format(t1))
    return df


def get_country_and_continent_code(country_name):
    """
    :param country_name:
    :return: Tuple : (code country, code continent)
    """
    t0 = time()
    # Traitement du code Pays
    try:
        cn_a2_code =  country_name_to_country_alpha2(country_name)
    except:
        cn_a2_code = 'Unknown'
        if country_name == 'United Kingdom':
            cn_a2_code = 'GB'
    # Traitement du code continent
    try:
        cn_continent = country_alpha2_to_continent_code(cn_a2_code)
    except:
        cn_continent = 'Unknown'
        if country_name == 'United Kingdom':
            cn_continent = 'EU'
    t1 = time() - t0
    print("get_continent in {0:.3f} secondes................... END".format(t1))
    return (cn_a2_code, cn_continent)


# 2. Get longitude and latitude
def get_geolocate(country, geolocator):
    """
    :param country: Typle : (code country, code continent)
    :param geolocator:
    :return: Typle : (latitude, longitude)
    """
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return LATITUDE_LONGITUDE.get(country, np.nan)


def complete_country_datas(countries_df,columns_name=['Country Code',"Continent", "Latitude","Longitude"], verbose=False):
    """
    :param df: DataFrame l'index est le nom du pays
    :param columns_name:
    :param verbose : Boolean - True pour mode debug
    :return: None
    """
    t0 = time()

    # Il faut ajouter codes, country, continent,
    if len(columns_name)<3:
        columns_name = ['Country Code',"Continent", "Latitude","Longitude"]

    for name in columns_name:
        countries_df[name] = ""

    geolocator = Nominatim(user_agent="catuserbot")
    for c, row in countries_df.iterrows():
        res = get_country_and_continent_code(c)
        if res[0] != 'Unknown' and res[1] != 'Unknown':
            countries_df.loc[c, columns_name[0]] = res[0]
            countries_df.loc[c, columns_name[1]] = res[1]

            geoloc = get_geolocate(res, geolocator)
            if geoloc != np.nan:
                try:
                    countries_df.loc[c, columns_name[2]] = geoloc[0]
                    countries_df.loc[c, columns_name[3]] = geoloc[1]
                except TypeError:
                    print("TypeError for :", res, c, geoloc)
            else:
                print("Country not found geoloc :", res, c)
        elif res[0] != 'Unknown':
            countries_df.loc[c, 'Country'] = res[0]
        elif res[1] != 'Unknown':
            countries_df.loc[c, 'Continent'] = res[1]
        else:
            if c == 'Holy See' or c == 'Vatican City':
                countries_df.loc[c, "Latitude"] = 41.902916
                countries_df.loc[c, "Longitude"] = 12.453389
            else:
                print("Country not known :", c)

    if verbose:
        print(countries_df)
        countries_df.to_csv("countries.csv")
