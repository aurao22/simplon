import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR

@ignore_warnings(category=ConvergenceWarning)
def logisticRegression_found_better_param(x, y, x_test, y_test, random_state=0, plot=False):
    res = {}
    better_score = 0
    better_penalty=0
    better_intercept=0
    better_n = 0

    # penalty
    penality_list = ['none', 'l2', 'l1', 'elasticnet']
    # fit_intercept
    fit_intercept_list = [True, False]
    # solver
    solver_list = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    i = 0
    for pen in penality_list:
        for inter in fit_intercept_list:
            for sol in solver_list:
                try:
                    model = LogisticRegression(penalty=pen, fit_intercept=inter, solver=sol,random_state=random_state)
                    model.fit(x, y)
                    score_test = model.score(x_test,y_test)
                    score_train = model.score(x,y)
                    res[i] = (model, score_test, score_train)
                    if score_test > better_score:
                        better_score = score_test
                        better_penalty = pen
                        better_intercept = inter
                        better_n = i
                    i += 1
                except ValueError as err:
                    print("{0}".format(err))
    print(f"{round(better_score,2)} de test, penalty {better_penalty},better_intercept:{better_intercept}, columns:{x.columns}")
    model, score_test, score_train = res[better_n]
    return model, score_test, score_train, better_n, better_penalty, better_intercept, res


def logisticRegression_found_better_old(X_train, X_test, y_train, y_test, random_state=0, plot=False):

    # on prend un maximum de colonne pour commencer
    to_predict_columns = 'Survived'
    
    columns_started = ['Pclass', 'sex_cod', 'title_cod', 'Age', 'family_on_board', 'Fare', 'embarked_cod', 'deck_cod']
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
                model, score_test, score_train, better_n, better_penalty, better_intercept, _ = logisticRegression_found_better_param(X_train[columns], y_train, X_test[columns], y_test, random_state, plot)
                
                # on stocke le résultat pour plus tard
                list_temp = test_res.get(score_test, {})
                list_temp[str_col] = (model, score_test, score_train, better_n, better_penalty, better_intercept)
                test_res[score_test] = list_temp

                ever_test.append(str_col)
                if score_test > better_score:
                    better_score = score_test
                    better_columns = columns
                    better_key = str(columns)
                    better_model = model
                    better_n_global = better_n
                    better_score_train = score_train
                    print(f"--------------------------------------------------------------------------------------------------------")
                    print(f"New Best :{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN = {better_n_global} :{better_key}")
                    print(f"--------------------------------------------------------------------------------------------------------")
                else:
                    print(f"{round(score_test,2)} de test <=> {round(score_train,2)} de train, KNN = {better_n} :{columns}")
            # On supprime une colonne
            columns.pop()
        #print(f"{round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN = {better_n_global} colonnes : {better_key} (started with:{save_init_col})")
    print(f"--------------------------------------------------------------------------------------------------------")
    print(f"Score {round(better_score,2)} de test <=> {round(better_score_train,2)} de train, KNN= {better_n_global} avec les colonnes : {better_key}")
    return better_model, better_columns, test_res, better_score



















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