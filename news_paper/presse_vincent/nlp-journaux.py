
import numpy as np
import pandas as pd
import urllib
import bs4 as bs
import pickle
import time
import mysql.connector
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer




###########################################
############# Execution MySQL #############
###########################################


def executeScriptsFromFile(file_name):
    """Fonction qui execute le scrip MySQL

    Args:
        file_name (string): Nom du fichier MySQL à executer
    """
    cnx = mysql.connector.connect(user='vincent', password='coucou22')
    cursor = cnx.cursor()
    fd = open(file_name, 'r')
    sqlFile = fd.read()
    fd.close()
    sqlCommands = sqlFile.split(';')
    for command in sqlCommands:
        try:
            if command.strip() != '':
                cursor.execute(command)
        except IOError as msg:
            print("Command skipped: ", msg)


###########################################


def get_page(url):
    """ 
    Fonction servant à récupérer une page web en fonction de son URL
    """
    req = urllib.request.Request(url)
    html = urllib.request.urlopen(req).read()
    return bs.BeautifulSoup(html, "lxml")


def fill_db_journaux(val):
    """ Création des journaux dans la BDD

    Args:
        val (array ou tuple): liste des journaux sous forme de tuple (nom, parution)
    """
    cnx = mysql.connector.connect(
        user='vincent', password='coucou22', database='projetjournaux')
    cursor = cnx.cursor()
    sql = "INSERT IGNORE INTO journal (nom, parution) VALUES (%s, %s)"

    if len(val) > 1:
        cursor.executemany(sql, val)
        cnx.commit()
        cursor.close()
        cnx.close()
    else:
        cursor.execute(sql, val)
        cnx.commit()
        cursor.close()
        cnx.close()

############### LE MONDE ###############


def get_summary_lemonde(page):
    """ 
    Permet de récupérer les menus présent sur le site Le Monde

    Args:
        page (_type_): il s'agit du return de la fonction get_page()

    Returns:
        array: Liste de string avec les menu du site Le Monde
    """
    summary = []
    print('Récupération des sommaires :')
    for item in tqdm(page.findAll('div', {'class': "wrapper"})):
        name = item.find("a").get('href')
        name = name[1:]
        summary.append(name)
    summary = summary[2:]
    print('-------------- fin du process get_summary_lemonde --------------')
    return summary


def get_articles_lemonde(summary, url):
    """ 
    Permet de récupérer les URL des articles des menus donnés sur le site Le Monde

    Args:
        summary (array): il s'agit du return de la fonction get_summary_lemonde()
        url (string): URL du site Le Monde

    Returns:
        array: tableau de tout les URL des articles à scrapper
    """
    data = []
    list_of_url = []
    print('Récupération des URL :')
    print('')
    for item in summary:
        url_summary = url + item
        list_of_url.append(url_summary)
    print('Récupération des articles :')
    print('')
    for url in tqdm(list_of_url):
        page = get_page(url)
        for item in page.findAll('div', {'class': "thread"}):
            name = item.find("a").get('href')
            data.append(name)
    print('longueur des data articles : ', len(data))
    print('-------------- fin du process get_articles --------------')
    return data


def article_to_db_lemonde(list_of_article):
    """Permet de scrapper les articles le site Le Monde

    Args:
        list_of_article (array):  il s'agit du return de la fonction get_articles_lemonde()

    Returns:
        array: tableau avec les (titres, content) de chaque article
    """
    data = []
    print('-------------------------------------')
    print('Création des articles (titres, content): ')
    print('-------------------------------------')
    for url in tqdm(list_of_article):
        page = get_page(url)
        text = ''
        article = []
        for title in page.findAll('h1', {'class': "article__title"}):
            article.append(title.getText())
        for content in page.findAll('p', {'class': "article__paragraph"}):
            t = content.getText()
            text += t
        if len(article) == 0:
            continue
        else:
            article.append(text)
            if len(article) == 2:
                data.append(article)
    return data


def fill_db_lemonde(data_lemonde):
    """Fonction servant à inseré dans la BDD les données relatif au journal 'Le Monde'.

    Args:
        data_lemonde (_type_): il s'agit du return de la fonction article_to_db_lemonde()
    """
    cnx = mysql.connector.connect(
        user='vincent', password='coucou22', database='projetjournaux')
    cursor = cnx.cursor()
    nom = "Le Monde"
    print('Insertion des articles de "Le Monde" en BDD')
    print('-------------------------------------------')
    for i in tqdm(data_lemonde):
        res_article = []
        if len(i) < 2:
            titre = 'non renseigné'
            content = i
        else:
            titre = i[0]
            content = i[1]
        if len(content) > 1:
            res_article.append((titre, content, nom))
            cursor.executemany(
                "INSERT IGNORE INTO articles VALUES (%s,%s,%s)", res_article)
        else:
            # print("article ", i[0], "non rajouté")
            continue
    cnx.commit()
    cursor.close()
    cnx.close()


def scrap_lemonde():
    """Fonction qui appel les fonctions précédement créer servant pour le site Le Monde
    """
    print('----------------------')
    print('Début du traitement : ')
    print('----------------------')
    temps1 = time.time()
    url = 'https://www.lemonde.fr/'
    page = get_page(url)
    summary = get_summary_lemonde(page)
    list_of_article = get_articles_lemonde(summary, url)
    data = article_to_db_lemonde(list_of_article)
    fill_db_lemonde(data)
    duration1 = time.time()-temps1
    print("Temps de traitement pour l'ajout en BDD des articles 'Le Monde' : ",
          "%15.2f" % duration1, "secondes")


############### ALLOCINE ###############


def get_summary_allocine(page):
    """ 
    Permet de récupérer les menus présent sur le site Allocine

    Args:
        page (_type_): il s'agit du return de la fonction get_page()

    Returns:
        _type_: Liste de string avec les menu du site Le Allocine
    """
    summary = []
    print('Récupération des sommaires :')
    for item in tqdm(page.findAll('li', {'class': "header-subnav-item"})):
        try:
            children = item.findChildren("a", recursive=False)
            for child in children:
                res = child.get('href')
                summary.append(res)
        except:
            print('------------------------------------')
            print("An exception occurred for ", item)
            pass
    list_of_exception = ['/news/cinema/', '/news/series/',
                         '/film/court-metrage/news/', '/podcast/']
    for url in list_of_exception:
        summary.append(url)
    print('-------------- fin du process get_summary_allocine --------------')
    return summary


def get_articles_allocine(summary):
    """ 
    Permet de récupérer les URL des articles des menus donnés sur le site Allocine

    Args:
        summary (_type_): il s'agit du return de la fonction get_summary_Allocine()
    

    Returns:
        _type_: tableau de tout les URL des articles à scrapper
    """
    url = 'https://www.allocine.fr'
    data = []
    list_of_url = []
    print('Récupération des URL :')
    print('')
    for item in summary:
        url_summary = url + item
        list_of_url.append(url_summary)
    # print(list_of_url)
    print('Récupération des articles :')
    print('')
    for url in tqdm(list_of_url):
        page = get_page(url)
        for item in page.findAll('h2', {'class': "meta-title"}):
            name = item.find("a").get('href')
            data.append(name)
    print('longueur des data articles : ', len(data))
    print('-------------- fin du process get_articles --------------')
    return data


def article_to_db_allocine(list_of_article):
    """Permet de scrapper les articles le site Allocine

    Args:
        list_of_article (_type_):  il s'agit du return de la fonction get_articles_allocine()

    Returns:
        _type_: tableau avec les (titres, content) de chaque article
    """
    data = []
    print('-------------------------------------')
    print('Création des articles (titres, content, auteur): ')
    print('-------------------------------------')
    for url in tqdm(list_of_article):
        url = 'https://www.allocine.fr' + url
        page = get_page(url)
        text = ''
        article = []
        for title in page.findAll('div', {'class': "titlebar-title titlebar-title-lg"}):
            article.append(title.getText())
        for content in page.findAll('p'):
            t = content.getText()
            text += t
        if len(article) == 0:
            continue
        else:
            article.append(text)
            if len(article) == 2:
                data.append(article)
    return data


def fill_db_allocine(data_allocine):
    """Fonction servant à inseré dans la BDD les données relatif au journal 'Allocine'.

    Args:
        data_lemonde (_type_): il s'agit du return de la fonction article_to_db_allocine()
    """
    try:
        cnx = mysql.connector.connect(
            user='vincent', password='coucou22', database='projetjournaux')
        cursor = cnx.cursor()
    except Exception as e:
        cursor.close()
        cnx.close()
        print(str(e))
    nom = "Allocine"
    print('Insertion des articles de "Allocine" en BDD')
    print('-------------------------------------------')
    for i in tqdm(data_allocine):
        res_article = []
        if len(i) < 2:
            titre = 'non renseigné'
            content = i
        else:
            titre = i[0]
            content = i[1]
        if len(content) > 1:
            res_article.append((titre, content, nom))
            cursor.executemany(
                "INSERT IGNORE INTO articles VALUES (%s,%s,%s)", res_article)
        else:
            continue
    cnx.commit()
    cursor.close()
    cnx.close()


def scrap_allocine():
    """Fonction qui appel les fonctions précédement créer servant pour le site Le Monde
    """
    url = 'https://www.allocine.fr/news/'
    page = get_page(url)
    summary = get_summary_allocine(page)
    list_of_article = get_articles_allocine(summary)
    data = article_to_db_allocine(list_of_article)
    fill_db_allocine(data)


###########################################


def make_df():
    """ Fonction qui récupérer l'ensemble des données présentes dans la BDD 'projetjournaux' - articles

    Returns:
        dataFrame: retourn une dataFrame avec les données présentes en BDD
    """
    print('Création de la DF depuis les données en BDD')
    temps1 = time.time()
    try:
        cnx = mysql.connector.connect(
            user='vincent', password='coucou22', database='projetjournaux')
        query = "Select * from articles;"
        result_dataFrame = pd.read_sql(query, cnx)
        cnx.close()
        filename = 'result_dataFrame.sav'
        pickle.dump(result_dataFrame, open(filename, 'wb'))
        duration1 = time.time()-temps1
        print("Sauvegarde réussi ! ")
        print('Nom du fichier créer pour la DF : ')
        print('')
        print(filename)
        print('')
        duration1 = time.time()-temps1
        print("Temps de traintement : ", "%15.2f" % duration1, "secondes")
    except Exception as e:
        cnx.close()
        print(str(e))
    return result_dataFrame


def cleanContent(review_text):
    stop_words = stopwords.words('french')
    new_stopwords_to_add = ['allociné', '892', '000', 'cookies']
    stop_words.extend(new_stopwords_to_add)
    review_text = str(review_text).lower().strip()
    review_text = word_tokenize(review_text)
    review_text = [word for word in review_text if word not in stop_words]
    lemma = WordNetLemmatizer()
    review_text = [lemma.lemmatize(word=w, pos='v') for w in review_text]
    review_text = [w for w in review_text if len(w) > 2]
    review_text = ' '.join(review_text)
    return review_text


def my_model(df):
    tfidf = TfidfVectorizer()
    mlp = MLPClassifier()
    X = tfidf.fit_transform(df['Clean_content']).toarray()
    y = df['journal']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    print("Entrainement du modèle MLPClassifier :")
    print("-------------------------------------")
    temps1 = time.time()
    mlp.fit(X_train, y_train)
    duration1 = time.time()-temps1
    print("Entrainement réussi !")
    print("Score : ", mlp.score(X_test, y_test))
    print("Temps de traintement : ", "%15.2f" % duration1, "secondes")
    print("-------------------------------------")
    print("Sauvegarde du modèle tfidf :")
    temps1 = time.time()
    filename2 = 'tfidf_model.sav'
    pickle.dump(tfidf, open(filename, 'wb'))
    duration1 = time.time()-temps1
    print("Sauvegarde du modèle :")
    temps1 = time.time()
    filename = 'finalized_model.sav'
    pickle.dump(mlp, open(filename, 'wb'))
    duration1 = time.time()-temps1
    print("Sauvegarde réussi ! ")
    print("Temps de traintement : ", "%15.2f" % duration1, "secondes")
    print('Nom du fichier créer pour le modèle prediction : ')
    print('')
    print(filename)
    print('')
    print('Nom du fichier créer pour le modèle tfidf : ')
    print('')
    print(filename2)
    print('')
    print("-------------------------------------")
    print('Traitement terminé.')
    print("-------------------------------------")
    xeval_numeric = tfidf.transform(test).toarray() 
    res = mlp.predict(xeval_numeric)
    print("pred : ", res)
    
test = ["Un goût amer. Emmanuel Macron a vivement dénoncé lundi 28 mars comme « une indignité » le fait qu'Éric Zemmour a laissé la foule scander « Macron assassin » sans réagir lors de son meeting au Trocadéro dimanche 27 mars, en conseillant au « candidat malentendant » de profiter de la réforme permettant le remboursement des prothèses auditives. « Il y a deux hypothèses : la première est l'indignité, c'est celle qui me semble la plus crédible, mais ce n'est pas une surprise », a déclaré le président candidat à son arrivée à Dijon pour une visite dans le cadre de la campagne pour la présidentielle. « La deuxième, c'est la méconnaissance d'une réforme très importante du quinquennat, c'est le 100 % santé. Maintenant, les prothèses auditives, les lunettes et les prothèses dentaires sont remboursées par la Sécurité sociale », a ajouté Emmanuel Macron. « Dix millions de Françaises et de Français y ont eu accès, cela fait partie de mon bilan et c'est un bilan social dont je suis fier. J'invite le candidat malentendant à pouvoir s'équiper à moindres frais. ». Des déclarations qui n'ont pas manqué de faire réagir Éric Zemmour dans la foulée sur BFM TV : « Il sait très bien ce qu'est un meeting, on n'entend pas tout ce qui se passe quand on est à la tribune. Ce sont des mauvais procès, on ne parle que d'une chose mineure […]. Ce ne sont pas des militants, ne mélangez pas les militants et spectateurs d'un meeting. On ne contrôle pas 100 000 personnes donc arrêtez de vous polariser volontairement sur un point mineur de ce meeting », a-t-il dit.",
        "Marvel en dévoile un peu plus sur son Doctor Strange in the Multiverse of Madness, grâce à de nouvelles photos.Alors que Spider-Man : No Way Home, le grand favori du prix des fans aux Oscars 2022, a été évincé par Zack Snyder et son Army of the Dead, Kevin Feige a dû être un peu déçu. Mais avec la Phase 4 déjà bien commencée, Marvel n'est toutefois pas près de perdre sa place de choix dans le coeur des fans (et du public tout court).Suite de Doctor Strange de Scott Derrickson, Doctor Strange in the Multiverse of Madness sera le cinquième film de la Phase 4 du MCU, initiée par Black Widow, et lancera la hype MCU au cinéma en 2022 (alors que Moon Knight s'occupera des séries dès le 30 mars). Le film se passera juste après les événements de Spider-Man : No Way Home, quelques jours-semaines après que le Sorcier a complètement détraqué le multivers.À en juger par la dernière bande-annonce de Doctor Strange in the Multiverse of Madness, il va falloir réparer tout ça. Et si tout le monde est sûrement plus à fond à l'idée de découvrir la bande-annonce de Avatar 2 lors de la sortie du film de Sam Raimi, Marvel a relancé la promotion de son Doctor Strange 2 avec des images de ses héros. Pour (encore) sauver le monde, Benedict Cumberbatch renfilera donc sa cape pour la sixième fois, mais il ne sera pas seul, son acolyte Wong, la puissante Wanda et la petite nouvelle America Chavez seront à ses côtés. Un trio qui devrait bien aider le Sorcier alors qu'il va sans doute devoir affronter de nombreuses entités dangereuses à travers les dimensions parallèles. Des antagonistes dont on ne connaît en revanche pas grand-chose.Alors qu'au niveau du casting, on sait que Chiwetel Ejiofor, Rachel McAdams et Michael Stuhlbarg seront aussi de la partie, et que Patrick Stewart, le célèbre professeur Xavier, devrait aussi être à l'affiche pour un rôle encore tenu secret, l'identité du méchant principal est encore très floue. Plusieurs méchants pourraient être dans Doctor Strange 2, mais difficile de savoir lequel sera le véritable ennemi de Stephen Strange, la promotion de Marvel n'en montrant pour le moment pas trop.",
        "Fabien Delrue et William Villeger (n°63 mondiaux) peuvent nourrir des regrets. Pour leur première apparition aux Championnats du monde, à Huelva (ESP), les deux Français ont été éliminés en huitièmes de finale par la paire japonaise Matsui-Takeuchi (n°43) 24-22, 25-23, après avoir eu deux volants de set dans la première manche, et un dans la deuxième. Delrue et Villeger s'étaient imposés au tour précédent face au duo britannique 17e mondial, Vendy-Lane (21-19, 12-21, 21-15).Déception également pour le double mixte. Exemptés du premier tour, vainqueurs des Danois Soby-Mikkelsen à la belle en 16es, Thom Gicquel et Delphine Delrue se sont inclinés ce jeudi face aux Japonais Matsutomo-Kaneko (n°18), 21-15, 21-16. En manque de compétition et de confiance en raison de la blessure de Thom Gicquel en octobre, les Français doivent en outre affronter des paires qui ont désormais étudié leur jeu. Giquel et Delrue vont désormais enchaîner sur une période d'entraînement.Tous les Français engagés en simple avaient été éliminés précédemment : Brice Leverdez et Qi Xuefei au premier tour, Thomas Rouxel et Marie Batomene au second.Les Mondiaux s'achèveront ce dimanche. ",
        "Pour un neuvième vol plané de chauve-souris, le Batman de Matt Reeves succède à celui de Tim Burton, Joel Schumacher ou Christopher Nolan avec cette fois Robert Pattinson dans le rôle-titre, Zoë Kravitz en Catwoman, Paul Dano très sphynx, Colin Farrell très pingouin, et même John Turturro. Trois heures pour suivre Bruce Wayne, alias Batman, dans son combat contre la corruption qui sévit à Gotham City. Tout cela en pleine campagne pour l'élection du nouveau maire et pour élucider les crimes du Sphinx. Un film où retentissent les plus grands troubles de l'époque : faillite démocratique, alerte terroriste, hantise de la catastrophe où gronde l'imminence d'un désastre…"]


if __name__ == '__main__':
    file_name = 'MySQL.sql'
    executeScriptsFromFile(file_name)
    val = [("Le Monde", "quotidien"), ("Allocine", "quotidien")]
    fill_db_journaux(val)
    scrap_lemonde()
    scrap_allocine()
    df = make_df()
    print('DF shape : ' ,df.shape)
    print('')
    print("Nombre d'article par journal : " ,df['journal'].value_counts())
    df['Clean_content'] = df['content'].apply(cleanContent)
    my_model(df)