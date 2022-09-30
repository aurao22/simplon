
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
import matplotlib.cm as cm

# ----------------------------------------------------------------------------------
#                        MODELS : Réseau de neurones
# ----------------------------------------------------------------------------------
import tensorflow as tf

def neural_network_with_categorical(x_train, y_train, x_test=None,  y_test=None, couche_activations=[(100,'sigmoid'), (10, 'softmax')], epochs=5,input_shape=None,dropout = None, verbose=0):
    nodes = []
    if input_shape:
        nodes.append(tf.keras.layers.Flatten(input_shape=input_shape))
        # tf.keras.layers.Flatten(input_shape=(28,28)),
    
    for nb_nodes, acti in couche_activations:
        nodes.append(tf.keras.layers.Dense(nb_nodes, activation=acti))
        # tf.keras.layers.Dense(100, activation='sigmoid'),
        # tf.keras.layers.Dense(10, activation='softmax')
        if dropout:
            tf.keras.layers.Dropout(dropout)
            # tf.keras.layers.Dropout(0.2),
    
    model = tf.keras.models.Sequential(nodes)
    
    y_categories = tf.keras.utils.to_categorical(y_train, num_classes=np.unique(y_train).shape[0])
    y_test_categories = tf.keras.utils.to_categorical(y_test, num_classes=np.unique(y_train).shape[0])
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # print(loss_fn(y_categories, predictions).numpy())
    
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])
    model.fit(x_train, y_categories, epochs=epochs)
    # model.fit(x_train, y_train, epochs=epochs)

    if x_test is not None and y_test is not None:
        print(f'\n----------------------------------------------------------------------------')
        print(f'{model.evaluate(x_test,  y_test_categories, verbose=verbose)} for {len(couche_activations)} couches ==> ', end='')
        for nb_nodes, acti in couche_activations:
            print(f'{nb_nodes} nodes on {acti}, ', end='')
        print(f'\n----------------------------------------------------------------------------')
        
    return model


# ----------------------------------------------------------------------------------
#                        MODELS : GridSearchCV
# ----------------------------------------------------------------------------------

@ignore_warnings(category=UserWarning)
def classifier_knn_grid(X_train, y_train, verbose=False, grid_params=None):
    if verbose: print("kneighborsclassifier", end="")
    if grid_params is None:
        grid_params = { 'kneighborsclassifier__n_neighbors': np.arange(1, 5),
                            'kneighborsclassifier__p': np.arange(1, 5),
                            'kneighborsclassifier__metric' : ['euclidean', 'manhattan'],
                            'kneighborsclassifier__algorithm' : ['auto'],
                            'kneighborsclassifier__metric_params' : [None],
                            'kneighborsclassifier__n_jobs' : [None],
                            'kneighborsclassifier__weights' : ['uniform']
                            }
    grid_pipeline = make_pipeline( KNeighborsClassifier())
    grid = GridSearchCV(grid_pipeline,param_grid=grid_params, cv=4)
    if X_train and y_train:
        grid.fit(X_train, y_train)

    if verbose: print("             DONE")
    return grid

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

# ----------------------------------------------------------------------------------
#                        MODELS : METRICS
# ----------------------------------------------------------------------------------
def get_metrics_for_the_model(model, X_test, y_test, y_pred,scores=None, model_name="", r2=None, full_metrics=False, verbose=0, transformer=None):
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
            av_list = ['micro', 'macro', 'weighted']
            if metric == 3:
                av_list.append(None)
            for average in av_list:
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

def get_metrics_for_model(model_dic, X_test, y_test, full_metrics=0, verbose=0):
    score_df = None
    scores = defaultdict(list)
    for model_name, (model, y_pred, r2) in model_dic.items():
        scores = get_metrics_for_the_model(model, X_test, y_test, y_pred, scores,model_name=model_name, r2=r2, full_metrics=full_metrics, verbose=verbose)

    score_df = pd.DataFrame(scores).set_index("Model")
    score_df.round(decimals=3)
    return score_df

from scipy.spatial.distance import cdist

def get_elbow_data(X, nb_clusters,random_state=0):
    distortions = []
    inertias = []
    for k in nb_clusters:
        kmeanModel = KMeans(n_clusters=k, random_state=random_state).fit(X)
            
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
    return distortions, inertias
# ----------------------------------------------------------------------------------
#                        MODELS : FIT AND TEST
# ----------------------------------------------------------------------------------
def fit_and_test_models(model_list, X_train, Y_train, X_test, Y_test, y_column_name=None, verbose=0, scores=None, metrics=0, transformer=None):
    
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
        try:
            model_name = mod_name
            if len(y_column_name) > 0:
                model_name = y_column_name+"-"+model_name

            if isinstance(model, LinearSVC):
                if ya.nunique() <= 2:
                    continue
            scores["Class"].append(y_column_name)
            scores["Model"].append(mod_name)
            md, score_l = fit_and_test_a_model(model,model_name, X_train, ya, X_test, yt, verbose=verbose, metrics=metrics, transformer=transformer) 
            modeldic[model_name] = md
            scorelist.append(score_l)
        except Exception as ex:
            print(mod_name, "FAILED : ", ex)
    
    for score_l in scorelist:
        for key, val in score_l.items():
            scores[key].append(val)    
    
    return modeldic, scores

@ignore_warnings(category=ConvergenceWarning)
def fit_and_test_a_model(model, model_name, X_train, y_train, X_test, y_test, verbose=0, metrics=0, transformer=None):
    t0 = time.time()
    if verbose:
        print(model_name, "X_train:", X_train.shape,"y_train:", y_train.shape, "X_test:", X_test.shape,"y_test:", y_test.shape)

    if transformer is not None:
        try:
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.fit_transform(X_test)
            if verbose:
                print(model_name, "After transform : X_train:", X_train.shape,"y_train:", y_train.shape, "X_test:", X_test.shape,"y_test:", y_test.shape)
        except:
            pass
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
        model_metrics = get_metrics_for_the_model(model, X_test, y_test, y_pred=None,scores=None, model_name=model_name, r2=r2, full_metrics=full, verbose=verbose, transformer=transformer)
        t_model = (time.time() - t0)   
        modeldic_score["metrics time"] = time.strftime("%H:%M:%S", time.gmtime(t_model))
        modeldic_score["metrics seconde"] = t_model

        for key, val in model_metrics.items():
            if "R2" not in key and "Model" not in key:
                modeldic_score[key] = val[0]

    return model, modeldic_score


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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.express as px

def draw_silhouette_curve(X, y, nb_clusters, random_state=42, verbose=0):
    
    silhouette_n_clusters = []

    for n_clusters in nb_clusters:
        # Entrainement du modèle
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(X)

        # Calcul du score silhouette
        silhouette_scr = round(silhouette_score(X, cluster_labels),2)
        if verbose:
            print(f"{n_clusters} clusters = {silhouette_scr} silhouette_score")

        silhouette_n_clusters.append(silhouette_scr)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # Représentation graphique
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        # Graphique 1
        ax1.set_title(f"The silhouette plot for the various clusters 0 to {n_clusters}.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_scr, color="red", linestyle="--")

        ax1.set_yticks([])  
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Graphique 2
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        
        ax2.scatter(X, y, marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # centroides
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Visualisation des clusters sur le dataset")

        plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
                    fontsize=14, fontweight='bold')
        plt.show()
    
    # Dernier graphe avec la silhouette
    fig = px.line(x=nb_clusters, y=silhouette_n_clusters , markers=True, title=f"Score silhouette par rapport au nombre de clusters")
    fig.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette score",
    )
    fig.update_layout(margin=dict(l=10, r=20, t=40, b=10))
    fig.update_xaxes(dtick=1)
    fig.show()

    return silhouette_n_clusters

def draw_silhouette_curve_old(df, X, nb_clusters, random_state=42, x_col = (5, "Score"), y_col=(6, "PIB"), verbose=0):
    
    silhouette_n_clusters = []

    for n_clusters in nb_clusters:
        # Entrainement du modèle
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(X)

        # Calcul du score silhouette
        silhouette_scr = round(silhouette_score(X, cluster_labels),2)
        if verbose:
            print(f"{n_clusters} clusters = {silhouette_scr} silhouette_score")

        silhouette_n_clusters.append(silhouette_scr)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # Représentation graphique
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        # Graphique 1
        ax1.set_title(f"The silhouette plot for the various clusters 0 to {n_clusters}.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_scr, color="red", linestyle="--")

        ax1.set_yticks([])  
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Graphique 2
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        
        ax2.scatter(df.iloc[::,x_col[0]], df.iloc[::,y_col[0]], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # centroides
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Visualisation des clusters sur le dataset")
        ax2.set_xlabel(x_col[1])
        ax2.set_ylabel(y_col[1])

        plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
                    fontsize=14, fontweight='bold')
        plt.show()
    
    # Dernier graphe avec la silhouette
    fig = px.line(x=nb_clusters, y=silhouette_n_clusters , markers=True, title=f"Score silhouette par rapport au nombre de clusters")
    fig.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette score",
    )
    fig.update_layout(margin=dict(l=10, r=20, t=40, b=10))
    fig.update_xaxes(dtick=1)
    fig.show()

    return silhouette_n_clusters

from matplotlib import colors as mcolors

def get_color_names():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    return sorted_names


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  c = 'red'
  if predicted_label == true_label:
    c = 'green'

  thisplot[true_label].set_color('blue')
  thisplot[predicted_label].set_color(c)

def plot_pred(x, y, predictions, range=range(0,1)):
    for i in range:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(i, predictions[i], y, x)
        plt.subplot(1,2,2)
        plot_value_array(i, predictions[i],  y|i)
        plt.show()


def plot_history(history, loss_name='loss', precision='accuracy', loss_val_name=None, precision_val=None):

    plt.figure(figsize=(18,5))
    plt.subplot(121)
    
    # Fonction de coût : Entropie croisée moyenne
    plt.plot(history.history[loss_name], c='steelblue', label='train')
    if loss_val_name is not None:
        plt.plot(history.history[precision_val], c='coral', label='validation')
    plt.title('Fonction de coût')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
    
    # Précision
    plt.subplot(122)
    plt.plot(history.history[precision], c='steelblue', label='train')
    if precision_val is not None:
        plt.plot(history.history[precision_val], c='coral', label='validation')
    plt.title('Précision')
    plt.legend(bbox_to_anchor=(1.05, 1),loc='best',  borderaxespad=0.)
    plt.show()