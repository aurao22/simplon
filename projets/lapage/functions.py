import matplotlib.pyplot as plt
import pandas as pd
from statistics import median
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR

def get_year_ca_by_month(df_ca, year='2021'):
    d1=df_ca.loc[year, 'ca'].resample('M').sum().reset_index()
    d1["month"] = d1["day"].dt.strftime('%m')
    d1 = d1.drop("day", axis=1)
    d1 = d1.rename(columns={"ca":year})
    d1 = d1.set_index("month")
    return d1


def category_prix(prix):
    """Retourne la catégorie en fonction du prix
    Args:
        prix (int)
    Returns:
        [String]: La catégorie
    """
    if 0 <= prix < 10:
        return "< 10 €"
    elif 10 <= prix < 25:
        return 'de 10 à 25 €'
    elif 25 <= prix < 100:
        pas = 25
        for i in range(25, 100, pas):
            if i <= prix < i+pas:
                return f'de {i} à {i+pas} €'
    elif 100 <= prix < 300:
        pas = 50
        for i in range(100, 300, pas):
            if i <= prix < i+pas:
                return f'de {i} à {i+pas} €'
    elif prix >= 300:
        return '>= 300 €'
    else:
        return np.nan

def create_sorter_prix():
    sorter = []
    for i in range (1,550,10):
        age = category_prix(i)
        if age not in sorter:
            sorter.append(age)
    return sorter

def category_age(age):
    if 0 <= age < 20:
        return '<20 ans'
    elif 20 <= age < 70:
        pas = 10
        for i in range(20, 70, pas):
            if i <= age < i+pas:
                return f'de {i} à {i+pas} ans'
    elif 70 <= age <= 120:
        return '+70 ans'
    else:
        return np.nan

def create_sorter_age():
    sorter = []
    for i in range (1,100,5):
        age = category_age(i)
        if age not in sorter:
            sorter.append(age)
    return sorter

print(create_sorter_age())

def categorie_frequence(frequence_sur_100_jours, base=100):
    # Base 100 = 12 mois
    # 
    freq2 = frequence_sur_100_jours / base
    limit = 0.66
    if freq2 > limit:
        return '> 8 x ans'
    elif 0.2 < freq2 <= limit:
        freq = int(round(freq2*12, 0))
        return f'{freq} x ans'
    else:
        return "<= 2 x ans"

def create_sorter_frequence():
    start= 100
    sorter = []
    i = start+10
    while i > 0:
        i -= 5
        age = categorie_frequence(i, start)
        if age not in sorter:
            sorter.append(age)
      
    return sorter

def create_df_frequence_by_customer_on_year(df, year):
    """[summary]

    Args:
        df (DataFrame): Dataframe
        year (str or int): Année à traiter

    Returns:
        [type]: [description]
    """
    age_freq_by_year = df[df['year']==year][["client_id", "day", "tranche_age", "age_reel", "montant_panier"]]
    # nombre de jour pour l'année
    nb_jours = max(age_freq_by_year["day"]) - min(age_freq_by_year["day"])
    nb_jours = nb_jours.n + 1
    nb_mois = nb_jours / 30
    
    age_freq_by_year_client = age_freq_by_year.groupby(["client_id", "tranche_age", "age_reel"]).count()
    age_freq_by_year_client = age_freq_by_year_client.rename(columns={"day": str(year)+"_nb_achats"})
    age_freq_by_year_client[str(year)+"_freq_moyenne"] = age_freq_by_year_client[str(year)+"_nb_achats"]/nb_mois

    # il faut en plus compter le montant_moyen du panier / Montant total
    df_client_temps_ca = df[df['year']==year][["client_id", "day", "tranche_age", "age_reel", "montant_panier"]]
    df_client_temps_ca = df_client_temps_ca.groupby(["client_id", "tranche_age", "age_reel"])["montant_panier"].agg(['sum', 'mean'])
    df_client_temps_ca = df_client_temps_ca.rename(columns={"sum": str(year)+"_cumul_montant", "mean": str(year)+"_mean_montant_panier"})
    res = pd.concat([age_freq_by_year_client, df_client_temps_ca], axis=1)
    res = res.drop("montant_panier", axis=1)

    return res


# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
def color_graph_background(ligne=1, colonne=1):
    figure, axes = plt.subplots(ligne,colonne)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    if isinstance(axes, np.ndarray):
        for axe in axes:
            # Traitement des figures avec plusieurs lignes
            if isinstance(axe, np.ndarray):
                for ae in axe:
                    ae.set_facecolor(PLOT_BAGROUNG_COLOR)
            else:
                axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    else:
        axes.set_facecolor(PLOT_BAGROUNG_COLOR)
    return figure, axes


def get_outliers_datas(df, colname):
    """[summary]

    Args:
        df ([type]): [description]
        colname ([type]): [description]

    Returns:
        (float, float, float, float): q_low, q_hi,iqr, q_min, q_max
    """
    # .quantile(0.25) pour Q1
    q_low = df[colname].quantile(0.25)
    #  .quantité(0.75) pour Q3
    q_hi  = df[colname].quantile(0.75)
    # IQR = Q3 - Q1
    iqr = q_hi - q_low
    # Max = Q3 + (1.5 * IQR)
    q_max = q_hi + (1.5 * iqr)
    # Min = Q1 - (1.5 * IQR)
    q_min = q_low - (1.5 * iqr)
    return q_low, q_hi,iqr, q_min, q_max


def graphe_outliers(df_out, column, q_min, q_max):
    """[summary]

    Args:
        df_out ([type]): [description]
        column ([type]): [description]
        q_min ([type]): [description]
        q_max ([type]): [description]
    """
    
    figure, axes = color_graph_background(1,2)
    # Avant traitement des outliers
    # Boite à moustaches
    #sns.boxplot(data=df_out[column],x=df_out[column], ax=axes[0])
    df_out.boxplot(column=[column], grid=True, ax=axes[0])
    # scatter
    df_only_ok = df_out[(df_out[column]>=q_min) & (df_out[column]<=q_max)]
    df_only_ouliers = df_out[(df_out[column]<q_min) | (df_out[column]>q_max)]
    plt.scatter(df_only_ok[column].index, df_only_ok[column].values, c='blue')
    plt.scatter(df_only_ouliers[column].index, df_only_ouliers[column].values, c='red')
    # Dimensionnement du graphe
    figure.set_size_inches(18, 7, forward=True)
    figure.set_dpi(100)
    figure.suptitle(column, fontsize=16)
    plt.show()

def draw_pie_multiple_by_value(df, column_name, values, compare_column_names, titre="", legend=True, verbose=False, max_col = 4 , colors=None):
    """ Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    nb_col = len(values)
    nb_row = 1
    if nb_col > max_col:
        more = 1
        if (nb_col % max_col) == 0:
            more = 0
        nb_row = (nb_col//max_col) + more
        nb_col = max_col

    figure, axes = color_graph_background(nb_row,nb_col)
    i = 0
    j = 0
    for val in values:
        ax = axes
        if nb_row == 1:
            ax = axes[i]
            i += 1
        else:
            ax = axes[i][j]
            j += 1
            if j == nb_col:
                i += 1
                j = 0
        _draw_pie(df[df[column_name]==val], compare_column_names, ax, colors=colors, legend=legend, verbose=verbose)
        ax.set_title(column_name+"="+str(val))
        ax.set_facecolor(PLOT_BAGROUNG_COLOR)   
        
    figure.set_size_inches(15, 5*nb_row, forward=True)
    figure.set_dpi(100)
    figure.suptitle(titre, fontsize=16)
    plt.show()
    print("draw_pie_multiple_by_value", column_name," ................................................. END")


def draw_pie_multiple(df, column_names, colors=None, verbose=False, legend=True):
    """Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        country_name_list (list[String], optional): Liste des pays à traiter. Defaults to [].
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    figure, axes = color_graph_background(1,len(column_names))
    i = 0
    for column_name in column_names:
        if len(column_names) > 1:
            _draw_pie(df, column_name, axes[i], colors=colors, legend=legend, verbose=verbose)
        else:
            _draw_pie(df, column_name, axes, colors=colors, legend=legend, verbose=verbose)
        i += 1
    figure.set_size_inches(15, 5, forward=True)
    figure.set_dpi(100)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    figure.suptitle(column_name+" REPARTITION", fontsize=16)
    plt.show()
    print("draw_pie", column_name," ................................................. END")


def draw_correlation_graphe(df, title, verbose=False):
    """Dessine le graphe de corrélation des données

    Args:
        df (DataFrame): Données à représenter
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    corr_df = df.corr()
    if verbose:
        print("CORR ------------------")
        print(corr_df, "\n")
    figure, ax = color_graph_background(1,1)
    figure.set_size_inches(15, 5, forward=True)
    figure.set_dpi(100)
    figure.suptitle(title, fontsize=16)
    sns.heatmap(corr_df, annot=True)
    plt.show()


def regLin_np(x, y):
    """
    Ajuste une droite d'équation a*x + b sur les points (x, y) par la méthode
    des moindres carrés.

    Args :
        * x (list): valeurs de x
        * y (list): valeurs de y

    Return:
        * a (float): pente de la droite
        * b (float): ordonnée à l'origine
    """
    # conversion en array numpy
    x = np.array(x)
    y = np.array(y)
    # nombre de points
    npoints = len(x)
    # calculs des parametres a et b
    a = (npoints * (x*y).sum() - x.sum()*y.sum()) / (npoints*(x**2).sum() - (x.sum())**2)
    b = ((x**2).sum()*y.sum() - x.sum() * (x*y).sum()) / (npoints * (x**2).sum() - (x.sum())**2)
    # renvoie des parametres
    return a, b

def draw_bar_tranches(df, group_columns=['tranche_age', 'categ'], sum_col='price', count_col='categ', unstack_col='categ', suptitle="Catégorie par tranche d'âge"):
    ca_categ_age = df.groupby(group_columns)[sum_col].sum().unstack(unstack_col).fillna(0)
    nb_categ_age = df.groupby(group_columns)[count_col].count().unstack(unstack_col).fillna(0)
    figure, axes = color_graph_background(2,1)

    # Affichage du nombre CA par catégorie
    ca_categ_age.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:,.0f} €". format(x)))
    axes[0].set_ylabel("CA en euros")
    axes[0].xaxis.set_visible(False)
    axes[0].grid(axis='y')

    # Affichage du nombre de livres
    nb_categ_age.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:,.0f}". format(x)))
    axes[1].set_ylabel("Nombre de livres")
    axes[1].grid(axis='y')

    figure.set_size_inches(16, 8, forward=True)
    figure.suptitle(suptitle, fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.show()

def draw_1(df_ca, ax, title):
    df_ca.loc["2021","ca"].plot(title=title, ax=ax)
    df_ca.loc["2021","ca"].resample("M").mean().plot(label="Moyenne par mois", lw=3, ls=":", alpha=0.8, ax=ax)
    df_ca.loc["2021","ca"].resample("W").mean().plot(label="Moyenne par semaine", lw=2, ls="--", alpha=0.8, ax=ax)
    df_ca.loc["2021","ca"].rolling(window=7, center=True).mean().plot(label="Moyenne mobile", lw=3, ls="-.", alpha=0.8, ax=ax)
    df_ca.loc["2021","ca"].ewm(alpha=0.6).mean().plot(label="EWM", lw=3, ls="--", alpha=0.8, ax=ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:,.0f} €". format(x)))
    ax.set_ylabel("CA en euros")
    ax.grid(axis='y')
    ax.legend()


def draw_2(df_ca, axe, title):
    df_ca.loc["2021","ca"].plot(title=title, ax=axe)
    # EWM = exponential weigthed function
    for i in np.arange(0.2, 1, 0.2):
        df_ca.loc["2021","ca"].ewm(alpha=i).mean().plot(label="EWM {:.1f}".format(i), lw=2, ls="--", alpha=0.8, ax=axe)
    axe.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:,.0f} €". format(x)))
    axe.set_ylabel("CA en euros")
    axe.grid(axis='y')
    axe.legend()

def draw_3(df, axe, title):
    ca_datas = df.loc["2021","ca"].resample("W").agg(["mean", "std", "min", "max"])
    ca_datas["mean"]["2021"].plot(label="moyenne par semaine", title=title, ax=axe)
    axe.fill_between(ca_datas.index, ca_datas["max"], ca_datas["min"], alpha=0.2, label="min-max par semaine")
    axe.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:,.0f} €". format(x)))
    axe.set_ylabel("CA en euros")
    axe.grid(axis='y')
    axe.legend()

def lorens(price, title, xlabel, ylabel ):
    #On place les observations dans une variable
    #Calcul de la somme cumulée et normalisation en divisant par la somme des observations
    lorenz_price = np.cumsum(price) / price.sum() 
    xmin = 1 - round(median(lorenz_price), 2)
    print(xmin)  
    figure, _ = color_graph_background(1,1)
    figure.set_size_inches(16, 8, forward=True)

    plt.plot(np.linspace(0,1,len(lorenz_price)), lorenz_price, drawstyle='steps-post', color='rosybrown', label='Lorenz')
    plt.fill_between(np.linspace(0,1,len(lorenz_price)) ,lorenz_price , color='#539ecd')
    plt.plot([0, 1], [0, 1], 'r-', lw=2, label='Distribution égalitaire')
    plt.vlines(x=xmin, ymin=0, ymax=.5, color='blue', linestyle='--', linewidth=1, label='Medial')
    plt.hlines(xmin=xmin, xmax=0, y=.5, color='blue', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.show()
    return lorenz_price

def _draw_pie(df, column_name, axe, colors=None, legend=True, verbose=False):
    """Fonction pour dessiner un graphe pie pour la colonne reçue

    Args:
        df (DataFrame): Données à représenter
        column_name (String): colonne à représenter
        axe ([type]): [description]
        colors ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): Mode debug. Defaults to False.
    """
    df_nova = df[~df[column_name].isna()][column_name].value_counts().reset_index()
    df_nova = df_nova.sort_values("index")
    # Affichage des graphiques
    axe.pie(df_nova[column_name], labels=df_nova["index"], colors=colors, autopct='%.0f%%')
    if legend:
        axe.legend(df_nova["index"], loc="upper left")
    else:
        legend = axe.legend()
        legend.remove()
    axe.set_title(column_name)
    axe.set_facecolor(PLOT_BAGROUNG_COLOR)