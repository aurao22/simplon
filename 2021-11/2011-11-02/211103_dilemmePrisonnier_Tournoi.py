import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')

from random import *
import DilemmePrisonnier as dilemme

# Le tournoi

liste = {}
strategie = {}
score = {}
duel = {}


# Ajouter les stratégies des joueurs ici
# ...

# Majorité, coopère seulement si l'autre joueur a coopéré en majorité.
def unSurDeux(liste_lui, liste_moi):
    if len(liste_moi) > 0:
        if liste_moi[-1]=='T':
            return 'C'
        else:  # premier coup
            return 'T'
    else:
        return dilemme.aleatoire(liste_lui, liste_moi)


def aurelie(liste_lui, liste_moi):
    res = dilemme.aleatoire(liste_lui, liste_moi)
    if len(liste_moi) > 0:
        mesT = liste_moi.count('T')
        if(mesT % 2 > 0):
            res = 'T'
    return res
liste['aurelie'] = []
strategie['aurelie'] = lambda lui, moi: aurelie(lui, moi)


def yohann(liste_lui, liste_moi):
    if 'T' not in liste_lui:
        return 'C'
    else:
        return 'T'
liste['yohann'] = []
strategie['yohann'] = lambda lui, moi: yohann(lui, moi)


def mehdi(liste_lui, liste_moi):
    if len(liste_moi) % 2:
        return "T"
    else:
        return dilemme.choice(dilemme.choix)
liste['Mehdi'] = []
strategie['Mehdi'] = lambda lui, moi: mehdi(lui, moi)

# Une chance
# coopere tant qu il n'a pas ete trahi 1 fois
def ellande(liste_lui, liste_moi):
    if len(liste_lui)>0:
        if liste_lui.count('T')>0:
            return 'T'
        else :
            return 'C'
    else :
        return 'C'
liste['ellande'] = []
strategie['ellande'] = lambda lui, moi: ellande(lui, moi)


def un_sur_trois(liste_lui, liste_moi):
    """ Trahi un coup sur trois, avec une trahison en premier coup !

    Args:
        liste_lui ([string]): liste des coups de l'adversaire.
        liste_moi ([string]): liste des coups effectués.

    Returns:
        T : trahit
        C : coopère
    """
    if len(liste_moi) % 3 == 0:
        return 'T'
    else:
        return 'C'
liste['vincent'] = []
strategie['vincent'] = lambda lui, moi: un_sur_trois(lui, moi)


def amaury(liste_lui, liste_moi):
    compteNombreT = 0
    if len(liste_lui) > 0:
        for i in liste_lui:
            if (i == "T"):
                compteNombreT += 1
        if (compteNombreT > 2):
            return 'T'
        else:
            return 'C'
    else:
        return 'C'


liste['amaury'] = []
strategie['amaury'] = lambda lui, moi: amaury(lui, moi)


def paul(liste_lui, liste_moi):
    # donnant donnant au début, suivi de pavlov (si il change, je change)
    if len(liste_lui) > 3:
        if liste_lui.count('T') > len(liste_lui) * 0.75: return 'T'  # pour contrer les trahisons perma
        if liste_moi[-1] == liste_lui[-1]:
            return liste_moi[-1]
        else:
            return liste_lui[-1]
    elif len(liste_lui) > 0:
        return liste_lui[-1]
    else:
        return 'C'


liste['Paul'] = []
strategie['Paul'] = lambda lui, moi: paul(lui, moi)


def halimata(liste_lui, liste_moi):
    if len(liste_lui) > 0 and liste_lui[0] == 'T':
        return 'T'
    else:
        return dilemme.choice(dilemme.choix)


liste['halimata'] = []

strategie['halimata'] = lambda lui, moi: halimata(lui, moi)


# Trahie au départ et sur les multiples de 3 et 8
def erwan(liste_lui, liste_moi):
    if len(liste_lui) > 0:
        if len(liste_lui) % 3 == 0:
            return 'T'
        elif len(liste_lui) % 8 == 0:
            return 'T'
        elif len(liste_lui) % 3 != 0 and len(liste_lui) % 8 != 0:
            return 'C'
    else:
        return 'T'


liste['Erwan'] = []
strategie['Erwan'] = lambda lui, moi: erwan(lui, moi)


def jeremy(liste_lui, liste_moi):
    listes = [dilemme.donnant_donnant(liste_lui, liste_moi), dilemme.majorite(liste_lui, liste_moi)]
    if len(liste_lui) > 0:
        if liste_lui[-1] == 'T':
            return 'T'
        elif liste_lui[-1] == 'C':
            return dilemme.choice(listes)
        else:
            return 'T'
    return 'C'


liste['jeremy'] = []
strategie['jeremy'] = lambda lui, moi: jeremy(lui, moi)


def damien(liste_lui, liste_moi):
    if len(liste_lui)%3==0:
            return 'C'
    else:
        return 'T'

liste['damien'] = []
strategie['damien'] = lambda lui, moi: damien(lui, moi)


def guillaume(liste_lui, liste_moi):  # NO MERCY, ON TRAHIT DIRECT !
    return 'T'

liste['guillaume'] = []
strategie['guillaume'] = lambda lui, moi: guillaume(lui, moi)


def julien(liste_lui, liste_moi):
    if len(liste_lui) > 0:
        if liste_lui[-1] == 'C':
            global choix
            return dilemme.choice(dilemme.choix)
        else:
            return 'T'
    else:
        return 'C'


liste['julien'] = []
strategie['julien'] = lambda lui, moi: julien(lui, moi)


def thomas(liste_lui, liste_moi):
    global nb_coups
    global nb_total_coups
    while nb_coups < (nb_total_coups / 2):
        if len(liste_lui) > 0:
            return liste_lui[-1]
        else:
            return 'C'
    else:
        return 'T'


liste['Thomas'] = []
strategie['Thomas'] = lambda lui, moi: thomas(lui, moi)


# Stratégie qui consiste à trahir dès que l’adversaire a trahi une fois
def claire(liste_lui, liste_moi):
    if 'T' not in liste_lui:
        return 'C'
    else:
        return 'T'


liste["Claire"] = []
strategie["Claire"] = lambda lui, moi: claire(lui, moi)


# stratégie basique: Trahit si l'adversaire a trahi au moins une fois lors des 3 essais précédents


def Anatole(liste_lui, liste_moi):
    if len(liste_lui) > 2:
        return liste_lui[-3]
    else:
        return 'T'
liste['Anatole'] = []
strategie['Anatole'] = lambda lui, moi: Anatole(lui, moi)


# ajouter des joueurs ci-dessous, selon les modèles des joueurs existants
# commencer ici
liste['Toujours seul'] = []
liste['Bonne poire'] = []
liste['Majorité'] = []
liste['Aléatoire'] = []
liste['Donnant donnant'] = []
liste['unSurDeux'] = []



# ...

strategie['Toujours seul'] = lambda lui, moi: dilemme.toujours_seul(lui, moi)
strategie['Bonne poire'] = lambda lui, moi: dilemme.bonne_poire(lui, moi)
strategie['Majorité'] = lambda lui, moi: dilemme.majorite(lui, moi)
strategie['Aléatoire'] = lambda lui, moi: dilemme.aleatoire(lui, moi)
strategie['Donnant donnant'] = lambda lui, moi: dilemme.donnant_donnant(lui, moi)
strategie['unSurDeux'] = lambda lui, moi: unSurDeux(lui, moi)

# ...
# terminer là

nb_total_coups = 5000 # à modifier

for joueur in liste.keys():
    score[joueur] = 0

for i in liste.keys():  # i et j sont les joueurs
    for j in liste.keys():
        liste[i] = []   # on recommence une partie
        liste[j] = []
        if i >= j:
            nb_coups = 0
            score_joueur1 = 0
            score_joueur2 = 0
            seed(45226)  # germe du générateur aléatoire
            while nb_coups < nb_total_coups:
                coup_joueur1 = strategie[i](liste[j], liste[i])
                coup_joueur2 = strategie[j](liste[i], liste[j])
                liste[i].append(coup_joueur1)
                if i != j:
                    liste[j].append(coup_joueur2)
                score_joueur1 += dilemme.gain(coup_joueur2, coup_joueur1)
                score_joueur2 += dilemme.gain(coup_joueur1, coup_joueur2)
                nb_coups += 1
            duel[(i, j)] = score_joueur1
            if i != j:
                duel[(j, i)] = score_joueur2
            score[i] += score_joueur1
            if i != j:
                score[j] += score_joueur2

# affichage des résultats


def trie_par_valeur(d):
    # retourne une liste de tuples triée selon les valeurs
    return sorted(d.items(), key=lambda x: x[1])


def trie_par_cle(d):
    # retourne une liste de tuples triée selon les clés
    return sorted(d.items(), key=lambda x: x[0])


name_size = 15

score_trie = trie_par_valeur(score)
score_trie.reverse()
print(nb_total_coups, "tirages")
for i in range(0, len(score_trie)):
    jName1 = score_trie[i][0] + (" " * (name_size - len(score_trie[i][0])))
    print(i, "-", jName1, ":", score_trie[i][1])


verbose = False

if verbose:
    duel_trie = trie_par_cle(duel)
    for i in range(0, len(duel_trie)):
        jName1 = duel_trie[i][0][0] + (" " * (name_size - len(duel_trie[i][0][0])))
        jName2 = duel_trie[i][0][1] + (" " * (name_size - len(duel_trie[i][0][1])))
        pts = duel_trie[i][1]
        if pts < 10:
            pts = " " + str(pts)
        print(jName1, "vs ", jName2, "gagne ", pts, "pts")

print(nb_total_coups, "tirages\nEND")
