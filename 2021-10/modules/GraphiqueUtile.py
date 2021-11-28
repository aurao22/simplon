import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')

import ListeUtil as myList
from random import sample
import seaborn as sns
import inspect
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ListeUtil as myList

# Propriétés utilisées pour les graphiques
colors = ["b", "g", "r", "c", "m", "y", "k"]
colorsNames = ["blue", "green", "red", "cyan", "magenta", "yellow"]

#ligneStyle = ["-", "--", ":", "-."]
ligneStyle = ['dashed', 'dashdot', 'dotted','solid', 'None']
#marqeurStyle = ["o", "x", "X", "--", "*", ".", ",", "V", "^", "<", ">", "1", "2", "3", "4", "s", "p", "h", "H", "+", "P", "d", "D", "|", "_"]
marqeurStyle = ["o", "x", "X", "*", ".",  "+", "_"]

colorsPaletteSeaborn = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]

combine = []


def displayGraph(df, x, y):
    c = getAColor()
    ls = getALigneStyle()
    ms = getAMarqeurStyle()

    plt.plot(df[x], df[y], color=c, marker=ms, linestyle=ls)
    plt.show()


def showCorrelationSeaborn(df, do_mask=False, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    if verbose:
        print("Function", functionName)
    else:
        print("Function", functionName, end='')

    # en cas d'erreur
    # htm_loc_map = loc_df.select_dtypes(["float64", "int64"])
    # corr = htm_loc_map.corr()
    # or
    # loc_df1.select_dtypes(exclude=['object', 'category'])
    corr_df = df.corr()
    if verbose:
        print("CORR ------------------")
        print(corr_df, "\n")

    if do_mask:
        mask = np.zeros_like(corr_df)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr_df, annot=True, mask=mask)
        # ou
        # ax = sns.heatmap(corr,mask = np.triu(np.ones_like(corr, dtype=bool)))
    else:
        sns.heatmap(corr_df, annot=True)
    plt.show()
    myList.logEnd("Function " + functionName, verbose)


def addGraphique(dataset):

    n = len(dataset)

    for key, i in zip(dataset.keys(), range(1, n+1)):
        found = False
        while(not found):
            c = getAColor()
            ls = getALigneStyle()
            ms = getAMarqeurStyle()
            style = c + ms + ls
            if style not in combine:
                combine.append(style)
                found = True
        # Définition du sous graphe
        plt.subplot(n, 1, i)
        # Paramétrage du sous graphique
        #plt.style.use('dark_background')
        plt.grid()
        plt.title(key)
        plt.plot(dataset[key],color=c, marker=ms, linestyle=ls)


def getAColor():
    c = sample(colors, 1)
    return c[0]


def getAColorName():
    c = sample(colorsNames, 1)
    return c[0]


def getAColorsPaletteSeaborn():
    c = sample(colorsPaletteSeaborn, 1)
    return c[0]


def getALigneStyle():
    ls = sample(ligneStyle, 1)
    return ls[0]


def getAMarqeurStyle():
    ms = sample(marqeurStyle, 1)
    return ms[0]


# Faire la moyenne des élèves par ligne
def addGraphRadarValues(legend, values, t, axes, maxValue, color, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')

    # Draw polygon representing values
    points = [(x, y) for x, y in zip(t, values)]
    points.append(points[0])
    points = np.array(points)
    codes = [path.Path.MOVETO, ] + \
            [path.Path.LINETO, ] * (len(values) - 1) + \
            [path.Path.CLOSEPOLY]
    _path = path.Path(points, codes)
    _patch = patches.PathPatch(_path, fill=True, color=color, linewidth=0, alpha=.1)
    axes.add_patch(_patch)
    _patch = patches.PathPatch(_path, fill=False, linewidth=2, color=color)
    res = axes.add_patch(_patch)
    res.set_label(legend)

    # Draw circles at value points
    plt.scatter(points[:, 0], points[:, 1], linewidth=2,
                s=50, color=color, edgecolor=color, zorder=maxValue)

    myList.logEnd("Function " + functionName)


# Affiche un radar avec les valeurs précisées
def showGraphRadar(title, properties, valuesList, maxValue, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName)

    # Use a polar axes
    axes = plt.subplot(111, polar=True)

    plt.legend = True
    # Set ticks to the number of properties (in radians)
    t = np.arange(0, 2 * np.pi, 2 * np.pi / len(properties))
    plt.xticks(t, [])

    # Set yticks from 0 to 10
    plt.yticks(np.linspace(0, maxValue, maxValue+1))

    # Draw polygon representing values
    colors = []
    keys = []
    for key, values in valuesList.items():
        color = getAColorName()
        keys.append(key)
        while color in colors:
            color = getAColorName()
        colors.append(color)
        addGraphRadarValues(key, values, t, axes, maxValue, color=color, verbose=verbose)
    axes.legend(loc='upper right')

    # Set axes limits
    plt.ylim(0, maxValue)

    # Draw ytick labels to make sure they fit properly
    for i in range(len(properties)):
        angle_rad = i / float(len(properties)) * 2 * np.pi
        angle_deg = i / float(len(properties)) * 360
        ha = "right"
        if angle_rad < np.pi / 2 or angle_rad > 3 * np.pi / 2: ha = "left"
        plt.text(angle_rad, maxValue+1, properties[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")

        # A variant on label orientation
        #    plt.text(angle_rad, 11, properties[i], size=14,
        #             rotation=angle_deg-90,
        #             horizontalalignment='center', verticalalignment="center")

    # Done
    plt.title(title)
    plt.show()
    myList.logEnd("Function " + functionName, True)


def displayBarGraphSeaborn(df, xName, yName, hueName, title, verbose=False):
    """ TODO
                Args:
                    df (DataFrame): le dataFrame à traiter
                    xName
                    yName
                    hueName
                    title
                    verbose (True or False): True pour mode debug
                Returns:
                    Nothing
                """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName)
    colors = sns.color_palette(myGraph.getAColorsPaletteSeaborn())
    g = sns.barplot(x=xName,y=yName, hue=hueName, data=df, palette=colors)
    g.set_ylabel(yName)
    g.set_xlabel(xName)
    plt.title(title)
    plt.show()
    myList.logEnd("Function " + functionName, verbose)


def displayBarGraphSeabornOneSerie(df, xName, title, verbose=False, show=True, isY=False):
    """ TODO
                Args:
                    df (DataFrame): le dataFrame à traiter
                    xName
                    yName
                    hueName
                    title
                    verbose (True or False): True pour mode debug
                Returns:
                    Nothing
                """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName)
    colors = sns.color_palette(getAColorsPaletteSeaborn())
    #sns.barplot(x=df["type_propriete"].value_counts().index, y=df["type_propriete"].value_counts())
    if not isY:
        g = sns.barplot(x=df[xName].value_counts().index,y=df[xName].value_counts(), palette=colors)
        g.set_ylabel("NB " + xName)
        g.set_xlabel(xName + " Index")
    else:
        g = sns.barplot(x=df[xName].value_counts(), y=df[xName].value_counts().index, palette=colors)
        g.set_xlabel("NB " + xName)
        g.set_ylabel(xName + " Index")
    plt.title(title)
    if show: plt.show()
    myList.logEnd("Function " + functionName, verbose)





def displayPieGraphSeaborn(df, label, values, title="", verbose=False):
    """ TODO
                    Args:
                        df (DataFrame): le dataFrame à traiter
                        xName
                        yName
                        hueName
                        title
                        verbose (True or False): True pour mode debug
                    Returns:
                        Nothing
                    """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName)
    # define Seaborn color palette to use
    colors = sns.color_palette(myGraph.getAColorsPaletteSeaborn())
    df_group = df.groupby(label).mean()
    print(df_group)
    labels = df_group.index
    values = df_group[values]
    # create pie chart
    plt.pie(values, labels=labels, colors=colors, autopct='%.0f%%')
    plt.title(title)
    plt.show()
    myList.logEnd("Function " + functionName, verbose)