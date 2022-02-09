
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
from collections import defaultdict
from sklearn.svm import LinearSVC


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output for binary classification."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0, 1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


@ignore_warnings(category=UserWarning)
def classifier_knn_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("kneighborsclassifier", end="")
    if grid_params is None:
        grid_params = { 'kneighborsclassifier__n_neighbors': np.arange(1, 20),
                            'kneighborsclassifier__p': np.arange(1, 10),
                            #'kneighborsclassifier__metric' : ['minkowski', 'euclidean', 'manhattan'],
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

@ignore_warnings(category=ConvergenceWarning)
def knn_fonction(X_train, Y_train, X_test, Y_test, y_column_name, n_neighbors=3, verbose=0):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train[y_column_name])
    score = knn.score(X_test, Y_test[y_column_name])
    if verbose:
        print("knn         :", round(score, 3))

    return knn, score

@ignore_warnings(category=ConvergenceWarning)
def linearSVC_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=0, verbose=0):
    model = LinearSVC(random_state=random_state)
    model.fit(X_train, Y_train[y_column_name])
    score = model.score(X_test, Y_test[y_column_name])
    if verbose:
        print("linearSVC  :", round(score, 3))

    return model, score

@ignore_warnings(category=UserWarning)
def classifier_logistic_grid(X_train, y_train, verbose=False, random_state=0, grid_params=None):
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

@ignore_warnings(category=ConvergenceWarning)
def logistic_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=0, verbose=0):
    logistic_model = LogisticRegression(random_state=random_state, verbose=verbose)
    logistic_model.fit(X_train, Y_train[y_column_name])
    score = logistic_model.score(X_test, Y_test[y_column_name])
    if verbose:
        print("logistic    :", round(score, 3))
    return logistic_model, score

@ignore_warnings(category=UserWarning)
def classifier_svc(X_train, y_train, random_state=0, grid_params=None, verbose=0):
    if verbose: print("SVC")
    #dict_keys(['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])
    if grid_params is None:
        grid_params = [
                {'kernel': ['rbf'], 'gamma': ['auto', 'scale', 0.1, 1, 10], 'C': [0.01, 0.1, 1.0, 10, 100]},
                {'kernel': ['poly'], 'degree': [3, 10, 30], 'C': [0.01, 0.1, 1.0, 10, 100]},
                {'kernel': ['linear'], 'C': [0.01, 0.1, 1.0, 10, 100]}
            ]

    clf = GridSearchCV(svm.SVC(random_state=random_state), grid_params, cv=4, n_jobs=4, verbose=verbose)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    if verbose: print("             DONE")
    return clf

@ignore_warnings(category=ConvergenceWarning)
def svc_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=0, verbose=0):
    svc = svm.SVC(random_state=random_state, verbose=verbose)
    svc.fit(X_train, Y_train[y_column_name])
    score = svc.score(X_test, Y_test[y_column_name])
    if verbose:
        print("SVC         :", round(score, 3))
    return svc, score

@ignore_warnings(category=UserWarning)
def classifier_svc_pca(X_train, y_train, random_state=0, grid_params=None, verbose=0):
    if verbose: print("SVC et PCA")

    pipe = Pipeline(steps=[('pca', PCA()), ('svm', svm.SVC(random_state=random_state))])

    # Syntaxe : nomdustep__nomduparamètre
    if grid_params is None:
        grid_params = {
            'pca__n_components': [2, 3, 4, 5, 15, 30, 45, 64],
            'svm__C': [0.01, 0.1, 1.0, 10, 100],
            'svm__kernel': ['rbf', 'poly', 'linear'],
            'svm__gamma': ['auto', 'scale', 0.1, 1, 10],
            'svm__degree': [3, 10, 30]
        }
    search = GridSearchCV(pipe, grid_params, n_jobs=4, verbose=1)
    search.fit(X_train, y_train)
    if verbose: print("             DONE")
    return search


def display_scores(models_list, X_test, y_test, X_test_pca=None, y_column_name=None):
   for model_name, model_grid in models_list.items():
        y_temp = y_test
        X_temp = X_test
        if y_column_name is not None:
            y_temp = y_temp[y_column_name]
        if "pca" in model_name and X_test_pca is not None:
            X_temp = X_test_pca
        
        print(model_name, " "*(18-len(model_name)), ":", round(model_grid.score(X_temp, y_temp), 3), end="")
        if isinstance(model_grid, GridSearchCV):
            print(model_grid.best_params_)
        else:
            print("")

from sklearn.metrics import *
from sklearn.metrics import roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay

from collections import defaultdict
import pandas as pd

def get_metrics_for_the_model(model, X_test, y_test, y_pred,scores=None, model_name="", r2=None, full_metrics=False, verbose=0):
    if scores is None:
        scores = defaultdict(list)
    scores["Model"].append(model_name)
        
    if r2 is None:
        r2 = round(model.score(X_test, y_test),3)
        
    if y_pred is None:
        t0 = time.time()
        y_pred = model.predict(X_test)
        t_model = (time.time() - t0)   
        # Sauvegarde des scores
        scores["predict time"].append(time.strftime("%H:%M:%S", time.gmtime(t_model)))
        scores["predict seconde"].append(t_model)
        
    scores["R2"].append(r2)
    scores["MAE"].append(mean_absolute_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    scores["MSE"].append(mse)
    scores["RMSE"].append(np.sqrt(mse))
    scores["Mediane AE"].append(median_absolute_error(y_test, y_pred))

    if full_metrics:
        try:
            y_prob = model.predict_proba(X_test)
        
            for metric in [brier_score_loss, log_loss]:
                score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
                try:
                    scores[score_name].append(metric(y_test, y_prob[:, 1]))
                except Exception as ex:
                    scores[score_name].append(np.nan)
                    if verbose > 0:
                        print("005", model_name, score_name, ex)
        except Exception as ex:
            if verbose > 0:
                print("003", model_name, "Proba", ex)
            scores['Brier  loss'].append(np.nan)
            scores['Log loss'].append(np.nan)
                
        for metric in [f1_score, recall_score]:
            score_fc_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            for average in [None, 'micro', 'macro', 'weighted']:
                try:
                    score_name = score_fc_name+str(average)
                    scores[score_name].append(metric(y_test, y_pred, average=average))
                except Exception as ex:
                    if verbose > 0:
                        print("005", model_name, score_name, ex)
                    scores[score_name].append(np.nan)

        # Roc auc  multi_class must be in ('ovo', 'ovr')   
        for metric in [roc_auc_score]:
            score_fc_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            for average in ['ovo', 'ovr']:
                try:
                    score_name = score_fc_name+str(average)
                    scores[score_name].append(metric(y_test, y_pred,multi_class= average))
                except Exception as ex:
                    if verbose > 0:
                        print("006", model_name, score_name, ex)
                    scores[score_name].append(np.nan)
    return scores

def get_metrics_for_model(model_dic, X_test, y_test, full_metrics=False, verbose=0):
    score_df = None
    scores = defaultdict(list)
    for model_name, (model, y_pred, r2) in model_dic.items():
        scores = get_metrics_for_the_model(model, X_test, y_test, y_pred, scores,model_name=model_name, r2=r2, full_metrics=full_metrics, verbose=verbose)

    score_df = pd.DataFrame(scores).set_index("Model")
    score_df.round(decimals=3)
    return score_df

from collections import defaultdict

def create_and_test_models(X_train, Y_train, X_test, Y_test, y_column_name=None, random_state=0, n_neighbors=3,verbose=0, scores=None, metrics=0):
    t0 = time()
    md_logistic, score_logistic = logistic_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=random_state,verbose=verbose)
    t_logistic = (time() - t0)/60
    t0 = time()
    md_svc, score_svc = svc_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=random_state,verbose=verbose)
    t_svc = (time() - t0)/60
    t0 = time()
    md_knn, score_knn = knn_fonction(X_train, Y_train, X_test, Y_test, y_column_name, n_neighbors=n_neighbors,verbose=verbose)
    t_knn = (time() - t0)/60
    md_linear_svc = None
    score_linear_svc = np.nan
    t_linear = 0
    if Y_test[y_column_name].nunique() > 2:
        t0 = time()
        md_linear_svc, score_linear_svc = linearSVC_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=random_state, verbose=verbose)
        t_linear = (time() - t0)/60

    # Sauvegarde des modèles entrainés
    modeldic = {}
    if y_column_name is None:
        y_column_name = ""
    modeldic[y_column_name+"-Logistic"] = md_logistic
    modeldic[y_column_name+"-SVC"] = md_svc
    modeldic[y_column_name+"-KNN"] = md_knn
    modeldic[y_column_name+"-Linear SVC"] = md_linear_svc

    # Sauvegarde des scores
    modeldic_score = {"Logistic-R2":score_logistic, "SVC-R2":score_svc, "KNN-R2":score_knn, "Linear SVC-R2":score_linear_svc,
     # Sauvegarde des temps d'exécution
                      "Logistic-seconde":t_logistic, "SVC-seconde":t_svc,"KNN-seconde":t_knn, "Linear SVC-seconde":t_linear}
    
    if scores is None:
        scores = defaultdict(list)
    # Sauvegarde des données
    for key, val in modeldic_score.items():
        if "seconde" in key:
            scores[key.replace("seconde", "time")].append(time.strftime('%H:%M:%S', time.gmtime(val))) 
        scores[key].append(val)    
    # Calcul et Sauvegarde des métriques
    if metrics > 0:
        full=metrics > 1
        for key, model in modeldic.items():
            model_name = key.split("-")[1]
            r2 = modeldic_score[model_name+"-R2"]
            
            model_metrics = get_metrics_for_the_model(model, X_test, Y_test[y_column_name], y_pred=None,scores=None, model_name=model_name, r2=r2, full_metrics=full, verbose=verbose)
            for key, val in model_metrics.items():
                if "Model" not in key:
                    scores[model_name+"-"+key].append(val[0])           

    return modeldic, scores

def get_empty_models_data(metrics=0):
    # Sauvegarde des scores
    modeldic_score = {"R2":np.nan,
                      "time":np.nan,
                      "seconde":np.nan,
                      }
    
    if metrics > 0:
        modeldic_score["MAE"] = np.nan,
        modeldic_score["MSE"] = np.nan,
        modeldic_score["RMSE"] = np.nan,
        modeldic_score["Mediane AE"] = np.nan,
    if metrics > 1:
        modeldic_score['Brier  loss'] = np.nan,
        modeldic_score['Log loss'] = np.nan,
        for metric in [f1_score, recall_score]:
            score_fc_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            for average in [None, 'micro', 'macro', 'weighted']:
                score_name = score_fc_name+str(average)
                modeldic_score[score_name] = np.nan,
        # Roc auc  multi_class must be in ('ovo', 'ovr')   
        for average in ['ovo', 'ovr']:
            score_name = "roc_auc_score"+str(average)       
            modeldic_score[score_name] = np.nan,

    return modeldic_score

def fit_and_test_models(model_list, X_train, Y_train, X_test, Y_test, y_column_name=None, verbose=0, scores=None, metrics=0):
    
    # Sauvegarde des modèles entrainés
    modeldic = {}
    yt = Y_test
    ya = Y_train
    # Sauvegarde des données
    if scores is None:
        scores = defaultdict(list)

    if y_column_name is None:
        y_column_name = ""
    else:
        yt = Y_test[y_column_name]
        ya = Y_train[y_column_name]
    
    scorelist = []
    for mod_name, model in model_list.items():
        model_name = mod_name
        if len(y_column_name) > 0:
            model_name = y_column_name+"-"+model_name

        if isinstance(model, LinearSVC):
            if y.nunique() <= 2:
                modeldic[model_name] = None
                score_linear_svc = get_empty_models_data(metrics=metrics)
                scorelist.append(score_linear_svc)
                continue
        scores["Class"].append(y_column_name)
        scores["Model"].append(mod_name)
        md, score_l = fit_and_test_a_model(model,model_name, X_train, ya, X_test, yt, verbose=verbose, metrics=metrics) 
        modeldic[model_name] = md
        scorelist.append(score_l)
    
    for score_l in scorelist:
        for key, val in score_l.items():
            scores[key].append(val)    
    
    return modeldic, scores

@ignore_warnings(category=ConvergenceWarning)
def fit_and_test_a_model(model, model_name, X_train, y_train, X_test, y_test, verbose=0, metrics=0):
    t0 = time.time()

    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    if verbose:
        print(model_name+" "*(20-len(model_name))+":", round(r2, 3))
    t_model = (time.time() - t0)
        
    # Sauvegarde des scores
    modeldic_score = {"Modeli":model_name,
                      "R2":r2,
                      "fit time":time.strftime("%H:%M:%S", time.gmtime(t_model)),
                      "fit seconde":t_model}
    
    # Calcul et Sauvegarde des métriques
    if metrics > 0:
        full=metrics > 1
        t0 = time.time()
        model_metrics = get_metrics_for_the_model(model, X_test, y_test, y_pred=None,scores=None, model_name=model_name, r2=r2, full_metrics=full, verbose=verbose)
        t_model = (time.time() - t0)   
        modeldic_score["metrics time"] = time.strftime("%H:%M:%S", time.gmtime(t_model))
        modeldic_score["metrics seconde"] = t_model

        for key, val in model_metrics.items():
            if "R2" not in key and "Model" not in key:
                modeldic_score[key] = val[0]

    return model, modeldic_score


def test_model_one_number_old(X_train, Y_train, X_test, Y_test, y_column_name=None, random_state=0, n_neighbors=3,verbose=0, scores=None, metrics=0):
    t0 = time()
    md_logistic, score_logistic = logistic_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=random_state,verbose=verbose)
    t_logistic = (time() - t0)/60
    t0 = time()
    md_svc, score_svc = svc_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=random_state,verbose=verbose)
    t_svc = (time() - t0)/60
    t0 = time()
    md_knn, score_knn = knn_fonction(X_train, Y_train, X_test, Y_test, y_column_name, n_neighbors=n_neighbors,verbose=verbose)
    t_knn = (time() - t0)/60
    md_linear_svc = None
    score_linear_svc = np.nan
    t_linear = 0
    if Y_test[y_column_name].nunique() > 2:
        t0 = time()
        md_linear_svc, score_linear_svc = linearSVC_fonction(X_train, Y_train, X_test, Y_test, y_column_name, random_state=random_state, verbose=verbose)
        t_linear = (time() - t0)/60

    # Sauvegarde des modèles entrainés
    modeldic = {}
    if y_column_name is None:
        y_column_name = ""
    modeldic[y_column_name+"-Logistic"] = md_logistic
    modeldic[y_column_name+"-SVC"] = md_svc
    modeldic[y_column_name+"-KNN"] = md_knn
    modeldic[y_column_name+"-Linear SVC"] = md_linear_svc
    
    if scores is None:
        scores = defaultdict(list)
    # Sauvegarde des scores
    scores["Column Name"].append(y_column_name)
    scores["Logistic-R2"].append(score_logistic)
    scores["SVC (OVO)-R2"].append(score_svc)
    scores["KNN-R2"].append(score_knn)
    scores["Linear SVC(OVR)-R2"].append(score_linear_svc)
    # Sauvegarde des temps d'exécution
    scores["Logistic time"].append(time.strftime('%H:%M:%S', time.gmtime(t_logistic)))
    scores["SVC-OVO time"].append(time.strftime('%H:%M:%S', time.gmtime(t_svc)))
    scores["KNN time"].append(time.strftime('%H:%M:%S', time.gmtime(t_knn)))
    scores["Linear SVC-OVR time"].append(time.strftime('%H:%M:%S', time.gmtime(t_linear)))
    # Si besoin de faire des tris
    scores["Logistic seconde"].append(t_logistic)
    scores["SVC-OVO seconde"].append(t_svc)
    scores["KNN seconde"].append(t_knn)
    scores["Linear SVC-OVR seconde"].append(t_linear)
    return modeldic, scores


def display_model_evaluation(model, X_test, y_test, y_pred,model_name="", r2=None, print_header = False, print_footer=False):
    emptyline = ["|----------------","|---------","|-----------","|--------","|------------","|----------","|"]
    if print_header:
        for col in emptyline:
            print(col, end="")
        print("")
        print("|Modèle          |R2       |MAE        |MSE     |RMSE        |Media AE  |")
        for col in emptyline:
            print(col, end="")
        print("")
    nb_space = len(emptyline[0])-1-len(model_name)
    print("|"+model_name+" "*nb_space, end="")

    if r2 is None:
        r2 = str(round(model.score(X_test, y_test),3))
    else:
        r2 = str(r2)
    print("|"+r2+" "*(len(emptyline[1])-1-len(r2)), end="")
    
    mae = str(round(mean_absolute_error(y_test, y_pred),3))
    print("|"+mae+" "*(len(emptyline[2])-1-len(mae)), end="")

    mse = mean_squared_error(y_test, y_pred)
    print("|"+str(mse)+" "*(len(emptyline[3])-1-len(str(mse))), end="")
    
    rmse = str(round(np.sqrt(mse),3))
    print("|"+rmse+" "*(len(emptyline[4])-1-len(rmse)), end="")

    mmae = str(round(median_absolute_error(y_test, y_pred),3))
    print("|"+mmae+" "*(len(emptyline[5])-1-len(mmae))+"|")

    if print_footer:
        for col in emptyline:
            print(col, end="")
        print("")

# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------

PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR


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


def draw_and_get_svm_svc(X_train, y_train, X_test=None, y_test=None, svc=None, kernel='rbf', C = 1.0, gamma="scale", h = None, xlabel=None, ylabel=None, title=None):

    if svc is None:
        svc = svm.SVC(kernel=kernel, C=C, gamma=gamma).fit(X_train, y_train)

    # Créer la surface de décision discretisée
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    # Pour afficher la surface de décision on va discrétiser l'espace avec un pas h
    if h is None:
        h = max((x_max - x_min) / 100, (y_max - y_min) / 100)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Surface de décision
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Afficher aussi les points d'apprentissage
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label="train", edgecolors='k', marker='+', cmap=plt.cm.coolwarm)
    if X_test is not None and y_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label="test", marker='*', cmap=plt.cm.coolwarm)

    if xlabel :
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title is None:
        title = ""
    plt.legend()
    title += f' SVC noyau {kernel}, C:{C}, gamma:{gamma}'
    plt.title(title)
    plt.show()
    return svc


def show_digit(some_digit, y):
    some_digit_image = some_digit.reshape(28, 28)
    color_graph_background(1,1)
    plt.imshow(some_digit_image, interpolation = "none", cmap = "afmhot")
    plt.title(y)
    plt.axis("off")


def draw_digits(df, y=None, nb=None):
    
    # plot some of the numbers
    if nb is None:
        nb = df.shape[0]

    nb_cols = 10
    nb_lignes = (nb//nb_cols)

    plt.figure(figsize=(14,(nb_lignes*1.5)))
    for digit_num in range(0,nb):
        plt.subplot(nb_lignes,nb_cols,digit_num+1)
        grid_data = df.iloc[digit_num].values.reshape(28,28)  # reshape from 1d to 2d pixel array
        plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")
        if y is not None:
            plt.title(y.iloc[digit_num])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

from sklearn.manifold import Isomap

def draw_predict(X, y, y_pred, title="", projection=None, c=None):
    if projection is None:
        iso = Isomap(n_components=2)
        projection = iso.fit_transform(X)
    figure, axe = color_graph_background()

    # Pour appliquer la couleur de y
    if c is None:
        c = y

    # ['viridis', 'cubehelix', 'plasma', 'inferno', 'magma', 'cividis']
    if y is not None:
        axe.scatter(projection[:, 0], projection[:, 1], label="Test", lw=0.2, c=c, cmap=plt.cm.get_cmap('viridis', 10))
        
    if y_pred is not None:
        axe.scatter(projection[:, 0], projection[:, 1], label="predict", marker='P', lw=0.5, c=c, cmap=plt.cm.get_cmap('viridis', 10))
    
    pcm = axe.pcolormesh(np.random.random((20, 20)) * 11, cmap=plt.cm.get_cmap('viridis', 10))
    figure.colorbar(pcm, ax=axe)
    plt.legend()
    figure.set_size_inches(10, 10, forward=True)
    figure.set_dpi(100)
    plt.title(title)


def draw_all_predict(X, y, y_pred, title="Représentation des prédictions", projection=None, c=None):
    if projection is None:
        iso = Isomap(n_components=2)
        projection = iso.fit_transform(X)

    # plot the results
    plt.figure(figsize=(18,15))

    figure, axes = color_graph_background(3,3)
    i = 0
    j = 0

    for digit in range(0,9):
        axe = axes[i][j]
        mask = y == digit

        # Pour appliquer la couleur de y
        c = y_pred[mask]
        x_digit = projection[mask]

        # ['viridis', 'cubehelix', 'plasma', 'inferno', 'magma', 'cividis']
        if y is not None:
            axe.scatter(x_digit[:, 0], x_digit[:, 1], label="Test", lw=0.2, c='b', cmap=plt.cm.get_cmap('viridis', 10))
        
        axe.scatter(x_digit[:, 0], x_digit[:, 1], label="predict", marker='P', lw=1, c=c, cmap=plt.cm.get_cmap('viridis', 10))
        axe.set_title(digit)
        #pcm = axe.pcolormesh(ticks=range(11), label='digit value', cmap=plt.cm.get_cmap('viridis', 10))
        
        axe.legend()
        pcm = axe.pcolormesh(np.random.random((20, 20)) * 11, cmap=plt.cm.get_cmap('viridis', 10))
        figure.colorbar(pcm, ax=axe)

        j += 1
        if j == 3:
            j = 0
            i += 1  
    
    figure.suptitle(title, fontsize=16)
    figure.set_size_inches(15, 20, forward=True)
    figure.set_dpi(100)
    plt.show()

import matplotlib as mat

def draw_PrecisionRecall_and_RocCurve(model, Y_test, y_score, model_name="SVC", colors=None):
    nb_lignes = 5
    nb_cols = 4
    ii = 0
    jj = 0
    figure, axes = color_graph_background(nb_lignes,nb_cols)
    if colors is None:
        colors = list(mat.colors.get_named_colors_mapping().values())
        
    for i in range(0,10):
        ax = axes[ii][jj]
        prec, recall, _ = precision_recall_curve(Y_test['class_'+str(i)], y_score[:,i], pos_label=model.classes_[1])
        PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax, color=colors[i])
        ax.set_title(str(i)+" - PrecisionRecall", fontsize=10)
        jj += 1
        if jj == nb_cols:
            jj = 0
            ii += 1 
        if ii < (nb_lignes-1):
            ax.get_xaxis().set_visible(False)
            ax.xaxis.set_ticklabels([])
        # -------------------------------------------------------------------------------------------
        ax = axes[ii][jj]
        fpr, tpr, _ = roc_curve(Y_test['class_'+str(i)], y_score[:,i], pos_label=model.classes_[1])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, color=colors[i])
        ax.set_title(str(i)+" - RocCurve", fontsize=10)
        jj += 1
        if jj == nb_cols:
            jj = 0
            ii += 1
        if ii < (nb_lignes-1):
            ax.get_xaxis().set_visible(False)
            ax.xaxis.set_ticklabels([])
    figure.suptitle(model_name+" PrecisionRecall and RocCurve", fontsize=16)
    figure.set_size_inches(15, 15, forward=True)
    figure.set_dpi(100)
    plt.show()

def draw_confusion(y_test, predictions_dic, verbose=0):
    nb_col = len(predictions_dic)
    figure, axes = color_graph_background(1, nb_col)
    i = 0
    for name, (model,pred) in predictions_dic.items():
        axe = axes[i]
        # Matrice de confusion
        cm = confusion_matrix(y_test, pred)
        if verbose:
            print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=axe)
        axe.set_title(name)
        i += 1
        if verbose:
            print(classification_report(y_test, pred))
    figure.set_size_inches(15, 5, forward=True)
    figure.set_dpi(100)
    plt.show()