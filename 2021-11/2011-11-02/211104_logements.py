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


# ----------------------------------------------------------------------------------------------------------
verbose = False

print("Chargement des données....")
file_name_logement = "211104_location_logements.csv"
loc_df = pd.read_csv(file_name_logement, sep=',')
myList.logEnd("Chargement des données " + file_name_logement, True)

# Partie 1 - Inspection des données et nettoyage
if verbose:
    myDf.displayInfo(loc_df)
    # Afficher les 5 premières lignes du dataframe
    print(loc_df.head(5))
    # Quelle est la taille du dataframe ?
    print("taille du dataframe", loc_df.shape)
    # Calculez les statistiques élémentaires (min, max, moyenne, écart type, etc) pour toutes les variables quantitatives.
    print("Statistiques élémentaires",loc_df.describe())
    # Calculez le taux moyen de valeurs manquantes pour chacune des variables.
    taux_moyen = loc_df.isna().sum(axis=0)/loc_df.shape[0]
    print("Taux moyen de valeurs manquantes\n",taux_moyen)

# Traitement des données
# Changez le type de la variable arrondissement en object.
loc_df = prepareDataLogement(loc_df, verbose)

# Vous allez vérifier que les quartiers et les arrondissements sont correctement associés.
# Pour cela, croisez les variables quartier et arrondissement dans un même tableau, vous devriez alors constater que toutes les locations d'un quartier sont situées dans un seul et même arrondissement.
# Indice : fonction crosstab de pandas.
cross = pd.crosstab(index=loc_df["quartier"],columns=loc_df["arrondissement"], margins=True, margins_name="Total")

if verbose:
    print(cross)

# Partie 2 - Analyse des données

# À l'aide d'un barplot, visualisez la répartition des types de propriétés.
if verbose:
    myGraph.displayBarGraphSeabornOneSerie(loc_df, 'type_propriete', "Répartition des types de propriétés", verbose )
    # À l'aide d'un histogramme faites ressortir la distribution des prix en fonction du type de propriété. Attention à configurer le nombre de bins et l'échelle log
    #sns.histplot(x=df["prix"], hue=df["type_propriete"], bins=30, log_scale=True)
    sns.histplot(data=loc_df, x="prix", hue="type_propriete", bins=30, log_scale=True, palette=myGraph.getAColorsPaletteSeaborn())
    plt.show()

    # À l'aide d'un barplot, visualisez la répartition des locations par quartier.
    # N'oubliez pas de trier les barres par ordre de grandeur afin d'améliorer l'interprétatibilité.
    myGraph.displayBarGraphSeabornOneSerie(loc_df, 'quartier', "Répartition des locations par quartier",
                                           verbose=verbose, isY=True)
    plt.show()

# On souhaite maintenant savoir quels sont les quartiers historiques de la plateforme.
# Donnez la valeur médiane de la variable nb_mois_en_activite en fonction du quartier.
df_quartier = loc_df.groupby(['quartier'])
df_quartier_median_activite = df_quartier['nb_mois_en_activite'].median().sort_values(ascending=True)
if verbose: print("valeur médiane de la variable nb_mois_en_activite en fonction du quartier\n", df_quartier_median_activite)

# Recherche des facteurs d'influence sur le prix de la location

# Discrétiser le variable note_moyenne en 4 sous-groupes de même tailles.
# Stockez les sous-catégories dans la nouvelle variable categories_note.
# Indice fonction qcut de pandas.
categories_note = pd.qcut(loc_df["note_moyenne"], q=5, duplicates="drop")
loc_df["categories_note"] = pd.qcut(loc_df["note_moyenne"], q=5, duplicates="drop")
if verbose:
    print("loc_df", loc_df["categories_note"].nunique())
    print("categories_note", categories_note.nunique())
    print(categories_note)
    print(loc_df["categories_note"])

# De la même manière, discrétisez la variable prix en 5 sous-groupes.
# Stockez les sous-catégories dans la nouvelle variable categories_prix.
categories_prix = pd.qcut(loc_df["prix"], q=5, duplicates="drop")
loc_df["categories_prix"] = pd.qcut(loc_df["prix"], q=5, duplicates="drop")
if verbose:
    print("loc_df", loc_df["categories_prix"].nunique())
    print("categories_prix", categories_prix.nunique())
    print("categories_prix.value_counts()\n", categories_prix.value_counts())
    print(categories_prix)
    print(loc_df["categories_prix"])

# À l'aide d'un tableau croisé des variables categories_note et categories_prix,
# vérifiez si il existe une influence du prix sur la note
if verbose:
    print(loc_df[["categories_note", "categories_prix"]])

cross_categorie = pd.crosstab(index=loc_df["categories_note"],columns=loc_df["categories_prix"], margins=True, margins_name="Total")
if verbose:
    print(cross_categorie)

# Approfondissez la question précédente avec un graphique displot, visualisez la distribution des prix en fonction des catégories de notes. N'oubliez pas l'échelle log pour améliorer la lisiblité.
if verbose:
    #sns.displot(hue=loc_df["categories_note"], x=loc_df["prix"], log_scale=True, kind="kde")
    sns.displot(data=loc_df, x="prix", hue="categories_note", log_scale=True, kind="kde")
    plt.show()

# Faites de même pour la variable lits en décomposant en fonction des sous-groupes de prix.
if verbose:
    sns.displot(data=loc_df, x="lits", hue="categories_prix", kind="kde")
    plt.show()


# Faites de même pour la variable nb_mois_en_activite.
# L'ancienneté de la location influence-t-elle le prix ?
if verbose:
    sns.displot(data=loc_df, x="nb_mois_en_activite", hue="categories_prix", kind="kde")
    plt.show()

# Visualisez maintenant la répartition des sous-catégories de notes en fonction du quartier (fonction countplot dans seaborn).
# Attention à bien trier les quartiers par ordre de grandeur en fonction du nombre de locations les mieux notées.
if verbose:
    sns.countplot(data=loc_df, hue="categories_note", y="quartier", order=loc_df['quartier'].value_counts(ascending=True).index)
    plt.show()

# Affichez la distribution des notes par quartier à l'aide d'un grouped boxplot.
# Attention à ne pas afficher les outliers (paramètre showfliers).
# Triez les boxplots en fonction de la valeur médiane.
if verbose:
    df_quartier = loc_df.groupby(['quartier'])
    df_quartier_median_note = df_quartier['note_moyenne'].median().sort_values(ascending=True)
    sns.boxplot(x=loc_df["note_moyenne"], y=loc_df["quartier"], showfliers=False, order=df_quartier_median_note.index)
    plt.show()

# On souhaite maintenant comprendre l'influence du quartier sur le prix de la location.
# À l'aide d'un grouped boxplot, visualisez la distribution des prix des locations en fonction du quartier.
# Attention à ne pas afficher les outliers (paramètre showfliers). Triez les boxplots en fonction de la valeur médiane.
if verbose:
    df_quartier_prix = df_quartier['prix'].median().sort_values(ascending=True)
    sns.boxplot(x=loc_df["prix"], y=loc_df["quartier"], showfliers=False, order=df_quartier_prix.index)
    plt.show()



# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


if verbose:
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(loc_df, row="quartier", hue="quartier", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "prix",
          bw_adjust=.5, clip_on=False, log_scale=True,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "prix", clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    g.map(label, "prix")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=.02)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True);
    plt.show()

# Analyse des corrélations linéaires
# Réaliser une analyse de la corrélation linéaire entre les variables quantitatives. Visualisez cela à l'aide d'une heatmap.
if verbose:
    myGraph.showCorrelationSeaborn(loc_df, do_mask=True, verbose=verbose)

# Analyse en Composantes Principales
# supprimer les 200 locations ayant les loyers les plus élevés.
loc_df_save = loc_df.copy()
if verbose:
    print(loc_df.head())
loc_df_clean = loc_df.sort_values("prix", ascending=False)
if verbose:
    print(loc_df_clean.head())
    print("loc_df_clean.shape - Before", loc_df_clean.shape)
loc_df_clean = loc_df_clean.drop(index=loc_df_clean.index[:200], axis=0)
if verbose:
    print("loc_df_clean.shape - AFTER", loc_df_clean.shape)


# Extrayez les variables nb_mois_en_activite, note_moyenne, capacite_accueil et prix.
# Puis standardisez ces variables avec StandardScaler de sklearn.

loc_df_filter = loc_df_clean.copy()
col_names = ["nb_mois_en_activite", "note_moyenne", "capacite_accueil", "prix"]
loc_df_filter = loc_df_filter[col_names]
# ------------
loc_df_filter_res = MinMaxScaler().fit_transform(loc_df_filter)
if verbose:
    print("MinMaxScaler\n", loc_df_filter_res)
    print(loc_df_filter_res.shape)
# -------------
scaler = StandardScaler()
f = scaler.fit(loc_df_filter)
m = scaler.mean_
if verbose:
    print(f)
    print(m)
# Centrage et Réduction
loc_df_filter_scaler = scaler.transform(loc_df_filter)
if verbose:
    print("StandardScaler\n", loc_df_filter_scaler)
    print(loc_df_filter_scaler.shape)

# selection des colonnes à prendre en compte dans l'ACP
data_pca = loc_df_filter[col_names]
# préparation des données pour l'ACP
data_pca = data_pca.fillna(data_pca.mean())
X = data_pca.values
names = loc_df_filter.index  # pour avoir les intitulés
features = loc_df_filter.columns

# Centrage et Réduction
std_scale = StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=len(col_names))
pca.fit(X_scaled)

if verbose:
    # Eboulis des valeurs propres
    myGraphPlay.display_scree_plot(pca)

# Cercle des corrélations
pcs = pca.components_
if verbose:
    myGraphPlay.display_circles(pcs, len(col_names), pca, [(0,1),(2,3),(4,5)], labels = np.array(features))
    # Affichez le cercle des corrélations des DEUX premières composantes.
    myGraphPlay.display_circles(pcs, 2, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))

verbose = True

#----------------------------------------------------------
if False:
    # selection des colonnes à prendre en compte dans l'ACP
    data_pca = loc_df_clean.copy()
    # préparation des données pour l'ACP
    data_pca = data_pca.fillna(data_pca.mean())
    X = data_pca.values
    names = loc_df_filter.index  # pour avoir les intitulés
    features = loc_df_filter.columns

    # Centrage et Réduction
    std_scale = StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)

    # Calcul des composantes principales
    pca = decomposition.PCA(n_components=2)
    pca.fit(X_scaled)

    # Cercle des corrélations
    pcs = pca.components_

# Projetez les points sur les deux premières composantes,
X_projected = pca.transform(X_scaled)

data_pc= pd.DataFrame(X_projected)
print(data_pc)
# Merge ou
data_pc["quartier"] = loc_df["quartier"]
print(data_pc)
data_pc["categories_prix"] = loc_df["categories_prix"]
print(data_pc)

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



