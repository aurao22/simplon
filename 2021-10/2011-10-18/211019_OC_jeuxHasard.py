#Question 3 :
# Soit 10.000 lancers d'un dé équilibré classique (à 6 faces). Parmi ces 10.000 lancers, on en prend 1.000 aléatoirement.
# On réalise 5 fois cette expérience, et on note m la moyenne des fréquences de 6 obtenus sur les 10.000 lancers et n la moyenne des fréquences de 4 obtenus dans le sous-échantillon.
# Quelles seraient les valeurs de m et n à l'issue de cette expérience ?

#Tuple représentant les faces d'un dé
import random
import statistics

de = (1, 2, 3, 4, 5, 6)

nbLancesTotal = 10000
resultatTousLances = []

for i in range(nbLancesTotal):
    resultatTousLances.append(random.choice(de))

moyennes = []
nbSixList = []
for i in range(5):
    sousListe = random.choices(resultatTousLances, 1000)
    nb6 = sousListe.count(6)
    print(nb6)
    nbSixList.append(nb6)
    mean = statistics.mean(sousListe)
    print(mean)
    moyennes.append(mean)











#----------------------------------------------------
#Question 4 :
#    Le jeu A est un jeu de pile ou face avec une pièce biaisée (pile avec une probabilité de p=0.49). On lance la pièce. Si l'on obtient pile, on gagne un euro, sinon on perd un euro.
#    Le jeu B est un jeu avec deux pièces biaisées. La pièce 1 donne pile avec une probabilité p1 = 0.09 et la pièce 2 donne pile avec une probabilité p2 = 0.74. Si la somme en jeu de K euros est un multiple de 3, on lance la pièce 1, sinon on lance la pièce 2. Comme dans le jeu A, si l'on obtient pile, on gagne un euro, sinon on perd un euro.
# Le jeu A est clairement perdant. Le jeu B l'est aussi (vous pourrez le vérifier). À présent, on va mixer les deux ! En effet, à chaque tour, on lance une pièce (cette fois-ci...) équilibrée ! Si l'on a pile, on joue au jeu A, sinon on joue au jeu B.

# On suppose que le joueur a 0 euros comme capital de départ.
# Après avoir joué 1.000.000 de parties, quel est le statut du jeu, du point de vue du joueur ?

