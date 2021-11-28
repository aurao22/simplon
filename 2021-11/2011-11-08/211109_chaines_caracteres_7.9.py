# Le mot le plus long
from time import time
import pprint


def dictionnaire_ordonne(light = False):
    # on lit le fichier et on range les mots alphabétiquement selon leur longueur
    fichier = open("211108_dico.txt", "r")
    dict_ord = {}
    for longueur in range(26):
        dict_ord[longueur + 1] = []
    mot = fichier.readline()

    while mot != '':
        mot = mot.strip('\n')
        if '-' in mot:
            mot = mot.replace('-', '')
        if "'" in mot:
            mot = mot.replace("'", '')
        dict_ord[len(mot)].append(mot)
        mot = fichier.readline()
        if light:
           try:
               for i in range(10):
                   mot = fichier.readline()
           except:
               break
    fichier.close()
    return dict_ord


def get_anagrammes(word, list_of_word, verbose=False):
    res = []
    word_set = sorted(word)

    for mot in list_of_word:
        mot_set = sorted(mot)
        if verbose:
            print("word_set", word_set, " - mot_set", mot_set)
        if mot_set == word_set:
            res.append(mot)
    return res

light = True
dico = dictionnaire_ordonne(light)
jouer = True
verbose = False

#while jouer:
if True:
    tirage = input('Entrez le nombre de lettres ou exit : ')
    jouer = not tirage == "exit"

    if jouer:
        taille = int(tirage)
        print(taille, ' lettres')
        t0 = time()

        mots_longueurs = dico[taille].copy()
        res = {}
        for mot in mots_longueurs:
            mots_longueurs.remove(mot)
            temp = get_anagrammes(mot, mots_longueurs, verbose)
            if len(temp)>0:
                res[mot] = temp
                # l'idée est d'enlever les mots qui ont déjà été traité en tant qu'anagramme
                # ceci pour réduire la liste de mots à traiter
                for w in res[mot]:
                    mots_longueurs.remove(w)

        t1 = time() - t0
        print(taille, 'Le script a mis', '{0:.3f}'.format(t1))

        for cle, valeurs in res.items():
            print(cle, valeurs)