'''
Exercice 6.8 - Le jeu des échelles
Règles du jeu
    Le joueur lance un dé et avance d'autant de cases que de points sur le dé.
    S'il tombe sur une case dans laquelle il y a le pied d'une échelle, il monte le long de celle-ci jusqu'en haut.
    S'il tombe sur une case dans laquelle il y a la tête d'un serpent, il doit redescendre jusqu'à la queue du serpent.
    La partie se termine quand on arrive sur la case 100.
    Si le joueur tire un dé qui ne lui permet pas d'arriver exactement sur la case 100, il recule du nombre de points supplémentaires sur le dé.
    Évidemment, une case ne peut être le départ ou l'arrivée que d'une seule échelle et un seul serpent.
Première question
    La question qui nous intéresse est la suivante : « Comment de fois faudra-t-il (en moyenne)
    lancer le dé pour terminer la partie ? » Simulez 100'000 parties sur le plateau ci-dessus.
Deuxième question (plus difficile)
    Déplacez (ou pas) horizontalement les serpents et les échelles du plateau ci-dessus de manière à :
    a) maximiser le nombre de lancers de dés
    b) minimiser le nombre de lancers de dés.
'''
import sys
import time

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import numpy as np
import random

SPECIALS_CASES = {1: 38, 4: 14, 9: 31, 17: 7, 28: 84, 21: 42, 54: 34, 51: 67, 64: 60, 62: 19,
                  71: 91, 80: 100, 87: 24, 93: 73, 95: 75, 98: 79}
FIRST_CASE = 0
LAST_CASE = 100


def getNextPosition(current_position, de, verbose=False):
    """ Calcule la prochaine position par rapport à la position actuelle, au tirage de dé et aux cases spéciales
            Args:
                current_position (int) : position actuelle du joueur
                de (int) Positif ou Négatif : tirage de dé ou le nombre à retirer en cas de dépassement du total
                verbose (True or False): True pour mode debug
            Returns:
                int : la nouvelle position du joueur
            """
    current_position += de
    if current_position == LAST_CASE:
        return current_position
    elif current_position > LAST_CASE:
        current_position = LAST_CASE - (current_position - LAST_CASE)
    current_position = SPECIALS_CASES.get(current_position, current_position)
    return current_position


def playGame(verbose=False):
    """ Joue une partie pour 1 joueur
        Args:
            verbose (True or False): True pour mode debug
        Returns:
            int : le nombre de lancés nécessaires pour terminer la partie
        """
    nb_de = 0
    current_position = FIRST_CASE
    while current_position != LAST_CASE:
        de = np.random.randint(1, 7)
        #de = random.randint(1, 6)
        nb_de += 1
        current_position = getNextPosition(current_position, de, verbose)
    return nb_de


# Comment de fois faudra-t-il (en moyenne) lancer le dé pour terminer la partie ?
start = time.time()
verbose = False
res = []
verbose = False
max = 100000
for i in range(max):
    nb_de = playGame(verbose)
    print(i, " Game ", nb_de, " lancer")
    res.append(nb_de)

moyenne = np.mean(res)
# 39.644 lancer moyen de dé pour 5000  parties
# 38.90756 lancer moyen de dé pour 7822
# 38.9601 lancer moyen de dé pour 10000  parties
# 38.93088 lancer moyen de dé pour 100000  parties
# 38.88788 lancer moyen de dé pour 50000  parties en  20.39232587814331  s
# 38.94422 lancer moyen de dé pour 100000  parties en  57.86154007911682  s
# 38.88301 lancer moyen de dé pour 100000  parties en  41.17047142982483  s
# 38.83721 lancer moyen de dé pour 100000  parties en  40.460960388183594  s
end = time.time()
elapsed = end - start
print(moyenne, "lancer moyen de dé pour", max, " parties en ", elapsed, " s")
