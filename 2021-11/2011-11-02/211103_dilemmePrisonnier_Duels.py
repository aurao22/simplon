# dilemme du prisonnier itéré, version duel
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')

from random import *
import DilemmePrisonnier as dilemme


# Une partie entre deux joueurs différents
liste = {}
score = {}

liste['Aléatoire'] = []
liste['Donnant donnant'] = []

for joueur in liste.keys():
    score[joueur] = 0

nb_coups = 0
nb_total_coups = 10  # à modifier

while nb_coups < nb_total_coups:
    coup_joueur1 = dilemme.aleatoire(liste['Donnant donnant'], liste['Aléatoire'])
    coup_joueur2 = dilemme.donnant_donnant(liste['Aléatoire'], liste['Donnant donnant'])
    liste['Aléatoire'].append(coup_joueur1)
    liste['Donnant donnant'].append(coup_joueur2)
    score['Aléatoire'] += dilemme.gain(coup_joueur2, coup_joueur1)
    score['Donnant donnant'] += dilemme.gain(coup_joueur1, coup_joueur2)
    nb_coups += 1

for joueur in liste.keys():
    jName = joueur + (" "*(20-len(joueur)))
    print("Score de", score[joueur], "pour", jName, "avec :", liste[joueur])

