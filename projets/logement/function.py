import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import itertools
from random import randint
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge, SGDRegressor, LinearRegression,HuberRegressor, QuantileRegressor,TheilSenRegressor 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

VALEUR_PONT_INCONNU = "X"
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR

@ignore_warnings(category=UserWarning)
def get_models_regression_logistic_grid(X_train, y_train, verbose=False, random_state=0, grid_params=None):
    if verbose: print("logisticregression")
    if grid_params is None:
        grid_params = { 'logisticregression__solver' : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        'logisticregression__penalty' : [None, 'l2', 'l1', 'elasticnet'],
                        'logisticregression__fit_intercept' : [True, False]}
    # penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None
    grid_pipeline = make_pipeline( LogisticRegression(random_state=random_state))
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid


@ignore_warnings(category=UserWarning)
def get_models_regression_linear_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("LinearRegression")
    if grid_params is None:
        grid_params = { 'linearregression__positive' :   [True, False],
                        'linearregression__normalize' :     [True, False],
                        'linearregression__fit_intercept' : [True, False]}
    grid_pipeline = make_pipeline( LinearRegression())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid

@ignore_warnings(category=UserWarning)
def get_models_regression_ridge_grid(X_train, y_train, verbose=False, random_state=0, grid_params=None):
    if verbose: print("Ridge")
    if grid_params is None:
        grid_params = { 'ridge__alpha' :         [1],
                        # 'ridge__normalize' :     [True, False], # Deprecated
                        'ridge__solver' :        ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                        'ridge__fit_intercept' : [True, False]}
    grid_pipeline = make_pipeline( Ridge(random_state=random_state))
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid

@ignore_warnings(category=UserWarning)
def get_models_regression_SGDRegressor_grid(X_train, y_train, verbose=False, random_state=0, grid_params=None):
    if verbose: print("SGDRegressor")
    if grid_params is None:
        grid_params = { 'sgdregressor__loss' :   ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'sgdregressor__penalty' : [None, 'l2', 'l1', 'elasticnet'],
                        'sgdregressor__fit_intercept' : [True, False]}
    grid_pipeline = make_pipeline( SGDRegressor(random_state=random_state))
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid

@ignore_warnings(category=UserWarning)
def get_models_regression_lasso(X_train, y_train, verbose=False, random_state=0, grid_params=None):
    if verbose: print("LassoCV")
    if grid_params is None:
        grid_params = { 'lassocv__alphas' :   [None],
                        # 'lasso__normalize' :     [True, False], => Deprecated
                        'lassocv__fit_intercept' : [True, False]}
    grid_pipeline = make_pipeline( LassoCV(random_state=random_state))
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid


@ignore_warnings(category=UserWarning)
def get_models_regression_random_forest(X_train, y_train, verbose=False, random_state=0, grid_params=None):
    if verbose: print("RandomForestRegressor")
    if grid_params is None:
        grid_params = { 'randomforestregressor__n_estimators' : np.arange(1, 100, 10),
                        'randomforestregressor__max_depth' : [None, 1, 2, 3],
                        'randomforestregressor__max_features' : ['auto', 'sqrt'],
                        'randomforestregressor__criterion' : ['squared_error', 'absolute_error', 'poisson'],
                        'randomforestregressor__min_samples_leaf' : [1],
                        'randomforestregressor__bootstrap' : [True, False],
                        'randomforestregressor__min_samples_split' : [1, 2, 3]}
    grid_pipeline = make_pipeline( RandomForestRegressor(random_state=random_state))
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid
    

@ignore_warnings(category=UserWarning)
def get_models_regression_knn_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("KNeighborsRegressor", end="")
    if grid_params is None:
        grid_params = { 'kneighborsregressor__n_neighbors': np.arange(1, 20),
                            'kneighborsregressor__p': np.arange(1, 10),
                            'kneighborsregressor__metric' : ['minkowski', 'euclidean', 'manhattan'],
                            'kneighborsregressor__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'kneighborsregressor__metric_params' : [None],
                            'kneighborsregressor__n_jobs' : [None],
                            'kneighborsregressor__weights' : ['uniform', 'distance']
                            }
    grid_pipeline = make_pipeline( KNeighborsRegressor())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid


@ignore_warnings(category=UserWarning)
def get_models_regression_outliers_robust_HuberRegressor_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("HuberRegressor", end="")
    if grid_params is None:
        grid_params = { 'huberregressor__fit_intercept' : [True, False]
                            }
    grid_pipeline = make_pipeline( HuberRegressor())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid

@ignore_warnings(category=UserWarning)
def get_models_regression_outliers_robust_QuantileRegressor_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("QuantileRegressor", end="")
    if grid_params is None:
        grid_params = { 'quantileregressor__quantile' : [0.5, 0.25, 0.75]
                            }
    grid_pipeline = make_pipeline( QuantileRegressor())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid

@ignore_warnings(category=UserWarning)
def get_models_regression_outliers_robust_QuantileRegressor_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("QuantileRegressor", end="")
    if grid_params is None:
        grid_params = { 'quantileregressor__quantile' : [0.5, 0.25, 0.75]
                            }
    grid_pipeline = make_pipeline( QuantileRegressor())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid


@ignore_warnings(category=UserWarning)
def get_models_regression_outliers_robust_HuberRegressor_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("HuberRegressor", end="")
    if grid_params is None:
        grid_params = { 'huberregressor__fit_intercept' : [True, False]
                            }
    grid_pipeline = make_pipeline( HuberRegressor())
    grid_knn = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid_knn.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid_knn


@ignore_warnings(category=UserWarning)
def get_models_grid_classifier_randomForest(X_train, y_train, verbose=False, random_state=0, grid_rf_params=None):
    if verbose: print("randomforestclassifier", end="")
    if grid_rf_params is None:
        grid_rf_params = { 'randomforestclassifier__criterion' : ["gini", "entropy"],
                       'randomforestclassifier__n_estimators' : np.arange(1, 100, 10)}
    grid_rf_pipeline = make_pipeline( RandomForestClassifier(random_state=random_state))
    grid_rf = GridSearchCV(grid_rf_pipeline,param_grid=grid_rf_params, cv=4)
    grid_rf.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid_rf

@ignore_warnings(category=UserWarning)
def get_models_classifier_knn_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("kneighborsclassifier", end="")
    if grid_params is None:
        grid_params = { 'kneighborsclassifier__n_neighbors': np.arange(1, 20),
                            'kneighborsclassifier__p': np.arange(1, 10),
                            'kneighborsclassifier__metric' : ['minkowski', 'euclidean', 'manhattan'],
                            'kneighborsclassifier__algorithm' : ['auto'],
                            'kneighborsclassifier__metric_params' : [None],
                            'kneighborsclassifier__n_jobs' : [None],
                            'kneighborsclassifier__weights' : ['uniform']
                            }
    grid_pipeline = make_pipeline( KNeighborsClassifier())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid
    

@ignore_warnings(category=UserWarning)
def get_models_classifier_decision_tree_grid(X_train, y_train, verbose=False, random_state=0, grid_dtc_params=None):
    if verbose: print("decisiontreeclassifier")
    if grid_dtc_params is None:
        grid_dtc_params = { 'decisiontreeclassifier__criterion' : ["gini", "entropy"],
                        'decisiontreeclassifier__splitter' : ["best", "random"]}
    grid_dtc_pipeline = make_pipeline( DecisionTreeClassifier(random_state=random_state))
    grid_dtc = GridSearchCV(grid_dtc_pipeline,param_grid=grid_dtc_params, cv=4)
    grid_dtc.fit(X_train, y_train)
    if verbose: print("             DONE")
    return grid_dtc
    

def get_models_grid(X_train, y_train, verbose=False):
    grid_dic = {}
    if verbose: print("randomforestclassifier", end="")
    grid_rf_params = { 'randomforestclassifier__criterion' : ["gini", "entropy"],
                   'randomforestclassifier__n_estimators' : np.arange(1, 100, 10)}
    grid_rf_pipeline = make_pipeline( RandomForestClassifier(random_state=random_state))
    grid_rf = GridSearchCV(grid_rf_pipeline,param_grid=grid_rf_params, cv=4)
    grid_rf.fit(X_train, y_train)
    grid_dic['randomforestclassifier'] = grid_rf
    if verbose: print(", kneighborsclassifier", end="")
    grid_params = { 'kneighborsclassifier__n_neighbors': np.arange(1, 20),
                        'kneighborsclassifier__p': np.arange(1, 10),
                        'kneighborsclassifier__metric' : ['minkowski', 'euclidean', 'manhattan'],
                        'kneighborsclassifier__algorithm' : ['auto'],
                        'kneighborsclassifier__metric_params' : [None],
                        'kneighborsclassifier__n_jobs' : [None],
                        'kneighborsclassifier__weights' : ['uniform']
                        }
    grid_pipeline = make_pipeline( KNeighborsClassifier())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    grid.fit(X_train, y_train)
    grid_dic['kneighborsclassifier'] = grid
    if verbose: print(", decisiontreeclassifier", end="")
    grid_dtc_params = { 'decisiontreeclassifier__criterion' : ["gini", "entropy"],
                    'decisiontreeclassifier__splitter' : ["best", "random"]}
    grid_dtc_pipeline = make_pipeline( DecisionTreeClassifier(random_state=random_state))
    grid_dtc = GridSearchCV(grid_dtc_pipeline,param_grid=grid_dtc_params, cv=4)
    grid_dtc.fit(X_train, y_train)
    grid_dic['decisiontreeclassifier'] = grid_dtc
    if verbose: print(", logisticregression", end="")
    grid_lr_params = { 'logisticregression__solver' : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    'logisticregression__penalty' : [None, 'l2', 'l1', 'elasticnet'],
                    'logisticregression__fit_intercept' : [True, False]}
    # penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None
    grid_lr_pipeline = make_pipeline( LogisticRegression(random_state=random_state))
    grid_lr = GridSearchCV(grid_lr_pipeline,param_grid=grid_lr_params, cv=4)
    grid_lr.fit(X_train, y_train)
    grid_lr.b
    grid_dic['logisticregression'] = grid_lr
    if verbose: print("                 DONE")
    return grid_dic

warnings.filterwarnings("ignore")
@ignore_warnings(category=ConvergenceWarning)
def found_better_model(X_train, X_test, y_train, y_test, models_grid_list, verbose=False):

    # on prend un maximum de colonne pour commencer
    columns_started = list(X_train.columns)
    better_grid_score_dic = {}
    better_grid_equals = {}
    ever_test = []
    
    # Modifier l'ordre des colonnes pour trouver encore d'autres configurations pertinentes
    # Positionnement de 6 suite aux tests lancés et des premiers résultats
    for subset in itertools.permutations(columns_started, 6):
        columns = list(subset)
            
        # a chaque tour, on regardera le meilleur score
        while len(columns)>0:
            str_col = str(sorted(columns))
            if str_col not in ever_test:
                for model_name, model_grid in models_grid_list.items():
                    model_grid.fit(X_train[columns], y_train)
                    score = model_grid.score(X_test[columns], y_test)

                    model_better_score = better_grid_score_dic.get(model_name, 0)
                    model_grig_res = (model_grid, score, str_col)
                    if score > model_better_score:
                        model_better_score = score
                        better_grid_equals[model_name] = [model_grig_res]
                        if verbose:
                            print(f"{model_name} New Best :{round(score,2)} de test, {str_col}, {model_grid.best_params_}")
                    elif score == model_better_score:
                        better_grid_equals[model_name].append(model_grig_res)
                        if verbose:
                            print(f"{model_name} Same Best :{round(score,2)} de test, {str_col}, {model_grid.best_params_}")

                    better_grid_score_dic[model_name] = model_better_score
                ever_test.append(str_col)
                if verbose>1: print(str_col, "         DONE")
            # On supprime une colonne
            columns.pop()
    
    return better_grid_score_dic, better_grid_equals


warnings.filterwarnings("ignore")
@ignore_warnings(category=ConvergenceWarning)
def found_better_config_by_model(X_train, X_test, y_train, y_test, verbose=False):

    # on prend un maximum de colonne pour commencer
    columns_started = list(X_train.columns)
    better_grid_score_dic = {}
    better_grid_equals = {}
    ever_test = []
    
    # Modifier l'ordre des colonnes pour trouver encore d'autres configurations pertinentes
    # Positionnement de 6 suite aux tests lancés et des premiers résultats
    for subset in itertools.permutations(columns_started, 6):
        columns = list(subset)
            
        # a chaque tour, on regardera le meilleur score
        while len(columns)>0:
            str_col = str(sorted(columns))
            if str_col not in ever_test:
                grid_dic = get_models_grid(X_train[columns], y_train)
                for model_name,grid in grid_dic.items():
                    score = grid.score(X_test[columns], y_test)

                    model_better_score = better_grid_score_dic.get(model_name, 0)
                    model_grig_res = (grid, score, str_col)
                    if score > model_better_score:
                        model_better_score = score
                        better_grid_equals[model_name] = [model_grig_res]
                        if verbose:
                            print(f"{model_name} New Best :{round(score,2)} de test, {str_col}, {grid.best_params_}")
                    elif score == model_better_score:
                        better_grid_equals[model_name].append(model_grig_res)
                        if verbose:
                            print(f"{model_name} Same Best :{round(score,2)} de test, {str_col}, {grid.best_params_}")

                    better_grid_score_dic[model_name] = model_better_score
                ever_test.append(str_col)
                if verbose>1: print(str_col, "         DONE")
            # On supprime une colonne
            columns.pop()
    
    return better_grid_score_dic, better_grid_equals





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
    # le risque  
    alpha = 0.05
    tab_contingence  = pd.crosstab(df[x],df[y])
    (chi2_val, p, degree, expected) = chi2_contingency(observed=tab_contingence )
    if debug:
        print('tableau de contingence : \n', tab_contingence)
        print(expected)
        critical = chi2.ppf(1-alpha, degree) 
        print('critical : ', critical)
    
    print(f"chi² val: {round(chi2_val,5)}, degree: {degree}, p value = {round(p, 5)} => ", end="") 
    # H0 : X et Y sont indépendantes
    if p <= alpha: # Rejet de l'hypothèse
        print(f'{y} dépend de {x} avec un risque {alpha}') 
    else: # Validation de l'hypothèse
        print(f'{y} NE dépend PAS de {x}')
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

def draw_y(X_test, x_col_name, y_test, y_pred, dict_y_pred):
    figure, axes = color_graph_background(2,3)
    i = 0
    j = 0
    for model_name, y_pred in dict_y_pred.items():
        if "SGDRegressor" in model_name:
            continue
        else:
            axe = axes[i][j]
            axe.scatter(X_test[x_col_name], y_test/1000, color='blue', label='expected')
            axe.scatter(X_test[x_col_name], y_pred/1000, color='red', marker='+', label='prediction')
            axe.set_title(model_name)
            axe.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:,.0f} K€". format(x)))
            axe.set_ylabel('median_house_value')
            axe.legend()
            j += 1
            if j == 3:
                j = 0
                i += 1
    figure.suptitle('Comparaison predictions vs expected > x='+x_col_name, fontsize=16)
    figure.set_size_inches(15, 5*3, forward=True)
    figure.set_dpi(100)
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


def draw_polynomiale(df, col_x, col_y, col_group=None):
    figure, axe = color_graph_background(1,1)

    
    # On affiche les données nettoyées
    df.plot.scatter(col_x, col_y, c=col_group, colormap='viridis', ax=axe)

    pf=PolynomialFeatures(degree=2,include_bias=False)
    Xpf= pf.fit_transform(X)

    linreg = LinearRegression()
    linreg.fit(Xpf,y)
    print(linreg.score(Xpf, y))

    X_new = np.linspace(min(X.values), max(X.values), X.shape[0]).reshape(len(X), 1)
    X_newPoly = pf.fit_transform(X_new)
    y_new = linreg.predict(X_newPoly)

    X_pred_poly = pf.fit_transform(X_pred)
    y_pred2 = linreg.predict(X_pred_poly)

    figure, axe = color_graph_background(1,1)
    axe.scatter(X, y)
    axe.plot(X_new, y_new, c='r', lw=3)
    axe.scatter(X_pred, y_pred2, c='g', marker='+', lw=3)
    figure.set_size_inches(16, 8, forward=True)

    axe.set_xlabel(col_x)
    axe.set_ylabel(col_y)


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


def draw_regression_3d2(df, col_x, col_y, col_z, col_group=None, colors=None): 
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection = '3d')

    # Pour faciliter la visualisation, on va changer la valeur de l'arrondissement (10)
    tmp_arr = df[col_z][:]
    c = list(colors.keys())
    p = None
    if col_group is None:
        p = ax.scatter(tmp_arr, df[col_x], df[col_y], c=tmp_arr, cmap="viridis")
    else:
        # df['continent'].map(colors)
        ax.scatter(tmp_arr, df[col_x], df[col_y], c=df[col_group].map(colors))
 
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_zlabel(col_z)
    if p is not None:
        fig.colorbar(p)
    plt.title(col_x + ", " + col_y + ", "+col_z)


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