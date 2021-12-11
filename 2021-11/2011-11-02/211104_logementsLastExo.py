import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ListeUtil as myList
import DataFrameUtil as myDf
import GraphiqueUtile as myGraph
import GraphiqueDisplayUtil as myGraphPlay
import inspect
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition


def prepareDataLogement(df, verbose=False):
    """ Prépare les données du Dataframe pour les logements
                Args:
                    df (DataFrame): le dataFrame à traiter
                    verbose (True or False): True pour mode debug
                Returns:
                    DataFrame : an updated DataFrame
                """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName)
    loc_df = df.copy()
    # Changez le type de la variable arrondissement en object.
    if verbose: print(loc_df.dtypes)
    loc_df = loc_df.astype({"arrondissement": 'object'})
    loc_df["arrondissement"] = loc_df["arrondissement"].astype('object')
    if verbose: print(loc_df.dtypes)

    # Remplacez les valeurs manquantes de la colonne lits par la valeur 1 (il y a normalement au moins un lit dans le bien loué).
    if verbose: print(loc_df.isna().sum(axis=0))
    loc_df.loc[loc_df['lits'].isna(), 'lits'] = 1

    # Remplacez les valeurs manquantes de la colonne chambres par la valeur médiane.
    loc_df = myDf.replaceNaNByMedianValue(loc_df, "chambres", verbose)
    if verbose: print(loc_df.isna().sum(axis=0))

    myList.logEnd("Function " + functionName, True)
    return loc_df


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


# ----------------------------------------------------------------------------------------------------------
verbose = False

print("Chargement des données....")
file_name_logement = "211104_location_logements.csv"
loc_df = pd.read_csv(file_name_logement, sep=',')
myList.logEnd("Chargement des données " + file_name_logement, True)

# Partie 1 - Inspection des données et nettoyage
# Traitement des données
# Changez le type de la variable arrondissement en object.
loc_df = prepareDataLogement(loc_df, verbose)

# Discrétiser le variable note_moyenne en 4 sous-groupes de même tailles.
# Stockez les sous-catégories dans la nouvelle variable categories_note.
# Indice fonction qcut de pandas.
categories_note = pd.qcut(loc_df["note_moyenne"], q=5, duplicates="drop")
loc_df["categories_note"] = pd.qcut(loc_df["note_moyenne"], q=5, duplicates="drop")

# De la même manière, discrétisez la variable prix en 5 sous-groupes.
# Stockez les sous-catégories dans la nouvelle variable categories_prix.
categories_prix = pd.qcut(loc_df["prix"], q=5, duplicates="drop")
loc_df["categories_prix"] = pd.qcut(loc_df["prix"], q=5, duplicates="drop")

loc_df = loc_df.sort_values("prix", ascending=False)
loc_df = loc_df.drop(index=loc_df.index[:200], axis=0)

# Extrayez les variables nb_mois_en_activite, note_moyenne, capacite_accueil et prix.
# Puis standardisez ces variables avec StandardScaler de sklearn.
col_names = ["quartier","nb_mois_en_activite", "note_moyenne", "capacite_accueil", "prix"]
# selection des colonnes à prendre en compte dans l'ACP
data_pca = loc_df.copy()[col_names]
# data_pca = loc_df[col_names]
# préparation des données pour l'ACP
data_pca = data_pca.fillna(data_pca.mean(numeric_only=True))
X = data_pca.values
names = data_pca.index  # pour avoir les intitulés
features = data_pca.columns

# Centrage et Réduction
std_scale = StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=2)
pca.fit(X_scaled)

if verbose:
    # Eboulis des valeurs propres
    myGraphPlay.display_scree_plot(pca)

# Cercle des corrélations
pcs = pca.components_
if verbose:
    # Affichez le cercle des corrélations des DEUX premières composantes.
    myGraphPlay.display_circles(pcs, 2, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))

verbose = True

# Projetez les points sur les deux premières composantes,
X_projected = pca.transform(X_scaled)

# puis isolez les valeurs par quartier et colorez par catégories de prix.
#X_projectedQuartier = X_projected.groupby(['quartier'])

# Indice : utilisez relplot de Seaborn.
# Projection des individus
# Create a visualization
if verbose:
    sns.relplot(data=data_pca,
        x=X_projected,
        # x=X_projected, y="tip", col="time",
        hue="quartier", style="smoker", size="size")
    plt.show(block=False)

print("END")



