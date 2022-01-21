import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import itertools
from random import randint
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


VALEUR_PONT_INCONNU = "X"
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR


def get_numeric_columns_names(df, verbose=False):
    """Retourne les noms des colonnes numériques
    Args:
        df (DataFrame): Données
        verbose (bool, optional): Mode debug. Defaults to False.

    Returns:
        List(String): liste des noms de colonne
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    return list(newdf.columns)


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


def calcul_chi2(df, x, y, debug=False):
    chi2_cross_age = pd.crosstab(df[x],df[y])
    (chi2_val, p, degree, expected) = chi2_contingency(observed=chi2_cross_age)
    if debug:
        print(chi2_cross_age)  
        print("chi² val:", chi2_val, "P value:", p, "degree:", degree)
        print(expected)
    alpha = 0.05
    print(f"chi² val: {chi2_val}, degree: {degree}, p value = {p} => ", end="") 
    if p <= alpha: 
        print(f'{y} ne dépend PAS de {x}') 
    else: 
        print(f'{y} dépend de {x}')
    return p


def knn_found_better_neigbors(x, y, x_test, y_test, knn_min=1, knn_max=10, metric = 'minkowski', p = 2, plot=False):
    res = {}
    better_score = 0
    better_n=0
    for i in range (knn_min, knn_max):
        model = KNeighborsClassifier(n_neighbors=i, metric = metric, p = p) 
        model.fit(x, y)
        score_test = model.score(x_test,y_test)
        score_train = model.score(x,y)
        res[i] = (model, score_test, score_train)
        if score_test > better_score:
            better_score = score_test
            better_n = i
    # print(f"{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN {better_n}, columns:{x.columns}")
    model, score_test, score_train = res[better_n]
    return model, score_test, score_train, better_n, res


def knn_found_better_config(X_train, X_test, y_train, y_test, knn_min=1, knn_max=10, plot=False, verbose=False):

    # on prend un maximum de colonne pour commencer
    columns_started = list(X_train.columns)
    ever_test = []
    better_score = 0
    better_score_train = 0
    better_n_global=0
    better_model = None
    better_columns = None
    better_key = None
    test_res = {}
    # train_res = {}
    # Modifier l'ordre des colonnes pour trouver encore d'autres configurations pertinentes
    # Positionnement de 6 suite aux tests lancés et des premiers résultats
    for subset in itertools.permutations(columns_started, 6):
        columns = list(subset)
                
        # a chaque tour, on regardera le meilleur score
        while len(columns)>0:
            str_col = str(sorted(columns))
            if str_col not in ever_test:
                model, score_test, score_train, better_n, _ = knn_found_better_neigbors(X_train[columns], y_train, X_test[columns], y_test, knn_min, knn_max, metric = 'minkowski', p = 2, plot=plot)
                
                # on stocke le résultat pour plus tard
                list_temp = test_res.get(score_test, {})
                list_temp[str_col] = (model, score_test, score_train, better_n)
                test_res[score_test] = list_temp

                # list_temp = train_res.get(score_train, {})
                # list_temp[str_col] = (model, score_test, score_train, better_n)
                # train_res[score_test] = list_temp

                ever_test.append(str_col)
                if score_test > better_score:
                    better_score = score_test
                    better_columns = columns
                    better_key = str(columns)
                    better_model = model
                    better_n_global = better_n
                    better_score_train = score_train
                    if verbose:
                        print(f"--------------------------------------------------------------------------------------------------------")
                        print(f"New Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN = {better_n_global} :{better_key}")
                elif score_test == better_score:
                    if verbose:
                        print(f"Same Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN = {better_n} :{str_col}")
                else:
                    if verbose>1:
                        print(f"{round(score_test,2)} de test <=> {round(score_train,2)} de train, KNN = {better_n} :{columns}")
            # On supprime une colonne
            columns.pop()
        #print(f"{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN = {better_n_global} colonnes : {better_key} (started with:{save_init_col})")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"KNN Score {round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN= {better_n_global} avec les colonnes : {better_key}")
    return better_model, better_key, test_res, better_score, better_score_train



def decisionTree_found_best(X_train, X_test, y_train, y_test, random_state=0, plot=False, verbose=False):
    # on prend un maximum de colonne pour commencer   
    columns_started = list(X_train.columns)

    # criterion
    criterion_list = ["gini", "entropy"]
    splitter_list = ["best", "random"]

    ever_test = []
    ever_error = []
    better_score = 0
    better_score_train = 0
    better_model = None
    better_columns = None
    better_param = None
    better_key = None
    test_res = {}
    
    # Modifier l'ordre des colonnes pour trouver encore d'autres configurations pertinentes
    # Positionnement de 6 suite aux tests lancés et des premiers résultats
    for subset in itertools.permutations(columns_started, 6):
        columns = list(subset)
                
        # a chaque tour, on regardera le meilleur score
        while len(columns)>0:
            str_col = str(sorted(columns))
            if str_col not in ever_test:
                for pen in criterion_list:
                    for inter in splitter_list:
                        str_param = f"criterion={pen}, splitter={inter},random_state={random_state}"
                        if str_param not in ever_error:
                            try:
                                model = DecisionTreeClassifier(criterion = pen, splitter=inter, random_state = random_state)
                                model.fit(X_train[columns], y_train)
                                score_test = model.score(X_test[columns], y_test)
                                score_train = model.score(X_train[columns], y_train)
                                if score_test > better_score:
                                    better_score = score_test
                                    better_penalty = pen
                                    better_intercept = inter
                                    better_score_train = score_train
                                    better_model = model
                                    better_columns = columns
                                    better_key = str_col
                                    better_param = str_param
                                    if verbose:
                                        print(f"--------------------------------------------------------------------------------------------------------")
                                        print(f"New Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {str_param} :{better_key}")
                                elif score_test == better_score:
                                    if verbose:
                                        print(f"Same Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {str_param} :{better_key}")
                                # on stocke le résultat pour plus tard
                                list_temp = test_res.get(score_test, {})
                                list_temp[str_col] = (model, score_test, score_train, better_penalty, better_intercept)
                                test_res[score_test] = list_temp
                                ever_test.append(str_col)
                            except ValueError as err:
                                # Prise en compte des cas de configuration de paramètre non compatible
                                if verbose:
                                    print("{0}".format(err))
                                ever_error.append(str_param)
                ever_test.append(str_col)
            # On supprime une colonne
            columns.pop()
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"DecisionTree Score {round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {better_param} :{better_key}")
    return better_model, better_key, test_res, better_score, better_score_train


def randomForest_found_best(X_train, X_test, y_train, y_test, random_state=0, plot=False, verbose=False):
    # on prend un maximum de colonne pour commencer   
    columns_started = list(X_train.columns)
    n_estimators_min = 1
    n_estimators_max = 100
    n_estimators_pas = 10
    # criterion
    criterion_list = ["gini", "entropy"]
    
    ever_test = []
    ever_error = []
    better_score = 0
    better_score_train = 0
    better_model = None
    better_columns = None
    better_param = None
    better_key = None
    test_res = {}
    
    # Modifier l'ordre des colonnes pour trouver encore d'autres configurations pertinentes
    # Positionnement de 6 suite aux tests lancés et des premiers résultats
    for subset in itertools.permutations(columns_started, 6):
        columns = list(subset)
        
        # a chaque tour, on regardera le meilleur score
        while len(columns)>0:
            str_col = str(sorted(columns))
            if str_col not in ever_test:
                for estimator in range(n_estimators_min, n_estimators_max, n_estimators_pas):
                    for inter in criterion_list:
                        str_param = f"n_estimators={estimator}, criterion={inter},random_state={random_state}"
                        if str_param not in ever_error:
                            try:
                                model = RandomForestClassifier(n_estimators = estimator, criterion = inter, random_state = random_state)
                                model.fit(X_train[columns], y_train)
                                score_test = model.score(X_test[columns], y_test)
                                score_train = model.score(X_train[columns], y_train)
                                if score_test > better_score:
                                    better_score = score_test
                                    better_penalty = estimator
                                    better_intercept = inter
                                    better_score_train = score_train
                                    better_model = model
                                    better_columns = columns
                                    better_key = str_col
                                    better_param = str_param
                                    if verbose:
                                        print(f"--------------------------------------------------------------------------------------------------------")
                                        print(f"New Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {str_param} :{better_key}")
                                elif score_test == better_score:
                                    if verbose:
                                        print(f"Same Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {str_param} :{better_key}")
                                # on stocke le résultat pour plus tard
                                list_temp = test_res.get(score_test, {})
                                list_temp[str_col] = (model, score_test, score_train, estimator, better_intercept)
                                test_res[score_test] = list_temp
                                ever_test.append(str_col)
                            except ValueError as err:
                                # Prise en compte des cas de configuration de paramètre non compatible
                                if verbose:
                                    print("{0}".format(err))
                                ever_error.append(str_param)
                ever_test.append(str_col)
            # On supprime une colonne
            columns.pop()
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"RandomForest Score {round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {better_param} :{better_key}")
    return better_model, better_key, test_res, better_score, better_score_train


@ignore_warnings(category=ConvergenceWarning)
def logisticRegression_found_better_full(X_train, X_test, y_train, y_test, random_state=0, plot=False, verbose=False):

    # on prend un maximum de colonne pour commencer   
    columns_started = list(X_train.columns)
    
    # penalty
    penality_list = ['none', 'l2', 'l1', 'elasticnet']
    # fit_intercept
    fit_intercept_list = [True, False]
    # solver
    solver_list = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]

    ever_test = []
    ever_error = []
    better_score = 0
    better_score_train = 0
    better_model = None
    better_columns = None
    better_param = None
    better_key = None
    test_res = {}
    
    # Modifier l'ordre des colonnes pour trouver encore d'autres configurations pertinentes
    # Positionnement de 6 suite aux tests lancés et des premiers résultats
    for subset in itertools.permutations(columns_started, 6):
        columns = list(subset)
                
        # a chaque tour, on regardera le meilleur score
        while len(columns)>0:
            str_col = str(sorted(columns))
            if str_col not in ever_test:
                for pen in penality_list:
                    for inter in fit_intercept_list:
                        for sol in solver_list:
                            str_param = f"penalty={pen}, fit_intercept={inter}, solver={sol},random_state={random_state}"
                            if str_param not in ever_error:
                                try:
                                    model = LogisticRegression(penalty=pen, fit_intercept=inter, solver=sol,random_state=random_state)
                                    model.fit(X_train[columns], y_train)
                                    score_test = model.score(X_test[columns], y_test)
                                    score_train = model.score(X_train[columns], y_train)
                                    if score_test > better_score:
                                        better_score = score_test
                                        better_penalty = pen
                                        better_intercept = inter
                                        better_score_train = score_train
                                        better_model = model
                                        better_columns = columns
                                        better_key = str_col
                                        better_param = str_param
                                        if verbose:
                                            print(f"--------------------------------------------------------------------------------------------------------")
                                            print(f"New Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {str_param} :{better_key}")
                                    elif score_test == better_score:
                                        if verbose:
                                            print(f"Same Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {str_param} :{better_key}")
                                    # on stocke le résultat pour plus tard
                                    list_temp = test_res.get(score_test, {})
                                    list_temp[str_col] = (model, score_test, score_train, better_penalty, better_intercept)
                                    test_res[score_test] = list_temp
                                    ever_test.append(str_col)
                                except ValueError as err:
                                    # Prise en compte des cas de configuration de paramètre non compatible
                                    if verbose>1:
                                        print("{0}".format(err))
                                    ever_error.append(str_param)
                ever_test.append(str_col)
            # On supprime une colonne
            columns.pop()
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"LogisticReg Score {round(better_score,2)} de test <=> {round(better_score_train,2)} de train, param = {better_param} :{better_key}")
    return better_model, better_key, test_res, better_score, better_score_train




# Fonction impliquant les modèles de  Machine Learning 
@ignore_warnings(category=ConvergenceWarning)
def best_other_model(X_train,y_train, X_test, y_test, more=False, verbose=False):

    # SVC method of svm class to use Support Vector Machine Algorithm
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, y_train)

    # SVC method of svm class to use Kernel SVM Algorithm
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, y_train)

    # Gauss
    gauss = GaussianNB()
    gauss.fit(X_train, y_train)
    
    # Résultats de chaque modèle.
    print(round(svc_lin.score(X_test, y_test), 2), "for test (",  round(svc_lin.score(X_train, y_train), 2), "for train) => Support Vector Machine (Linear)")
    print(round(svc_rbf.score(X_test, y_test), 2), "for test (",  round(svc_rbf.score(X_train, y_train), 2), "for train) => Support Vector Machine (RBF)")
    print(round(gauss.score(X_test, y_test), 2), "for test (",  round(gauss.score(X_train, y_train), 2), "for train) => Gaussian Naive Bayes")
    return svc_lin, svc_rbf, gauss


@ignore_warnings(category=ConvergenceWarning)
def best_model2(X_train,y_train, X_test, y_test, more=False, verbose=False):

    # Logistic Regression 
    log, log_better_columns, _, log_better_score, log_better_score_train = logisticRegression_found_better_full(X_train, X_test, y_train, y_test, random_state=0, plot=False, verbose=False)
    
    # KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
    knn, knn_better_columns, _, knn_better_score, knn_better_score_train = knn_found_better_config(X_train,X_test, y_train, y_test, verbose=False)
    
    # SVC method of svm class to use Support Vector Machine Algorithm
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, y_train)

    # SVC method of svm class to use Kernel SVM Algorithm
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, y_train)

    # Gauss
    gauss = GaussianNB()
    gauss.fit(X_train, y_train)

    # Arbre de décision
    tree, tree_better_columns, _, tree_better_score, tree_better_score_train = decisionTree_found_best(X_train, X_test, y_train, y_test, random_state=0, plot=False, verbose=False)
    
    # Forêt aléatoire
    forest, forest_better_columns, forest_test_res, forest_better_score, forest_better_score_train = randomForest_found_best(X_train, X_test, y_train, y_test, random_state=0, plot=False, verbose=False)
    
    # Résultats de chaque modèle.
    print('[0]Logistic Regression Training Accuracy:', log_better_score, "( train:", log_better_score_train, ") with:", log_better_columns)
    print('[1]K Nearest Neighbor Training Accuracy:', knn_better_score, "( train:", knn_better_score_train, ") with:", knn_better_columns)
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_test, y_test), "( train:", svc_lin.score(X_train, y_train), ")")
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_test, y_test), "( train:", svc_rbf.score(X_train, y_train), ")")
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_test, y_test), "( train:", gauss.score(X_train, y_train), ")")
    print('[5]Decision Tree Classifier Training Accuracy:', tree_better_score, "( train:", tree_better_score_train, ") with:", tree_better_columns)
    print('[6]Random Forest Classifier Training Accuracy:', forest_better_score, "( train:", forest_better_score_train, ") with:", forest_better_columns)

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest

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


def draw_regression2(df, col_x, col_y, col_group=None):
    figure, axe = color_graph_background(1,1)

    plt.xlabel(col_x)
    plt.ylabel(col_y)
    # On affiche les données nettoyées
    df.plot.scatter(col_x, col_y, c=col_group, colormap='viridis', ax=axe)

    mini = min(df[col_x])
    maxi = max(df[col_x]) + 1

    X = np.matrix([np.ones(df.shape[0]), df[col_x]]).T
    y = np.matrix(df[col_y]).T
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    axe.plot([mini,maxi], [theta.item(0),theta.item(0) + maxi * theta.item(1)], linestyle='--', c='#000000')

    figure.set_size_inches(16, 8, forward=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(col_x + " " + col_y)
    plt.show()


def draw_regression_3d(df, col_x, col_y, col_hue, col_group=None): 
    fig = plt.figure(figsize=(16, 8)).gca(projection='3d')
    # Pour faciliter la visualisation, on va changer la valeur de l'arrondissement (10)
    tmp_arr = df[col_hue][:]
    if col_group is None:
        fig.scatter(tmp_arr, df[col_x], df[col_y], c=tmp_arr, cmap="viridis")
    else:
        fig.scatter(tmp_arr, df[col_x], df[col_y], c=df[col_group], cmap="viridis")
 
    plt.xlabel(col_hue)
    plt.ylabel(col_x)
    plt.xticks(rotation=45, ha="right")
    plt.title(col_x + " " + col_y)
    plt.legend()
    plt.show()


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