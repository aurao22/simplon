# ###
# Premier petit projet python: Devine le nombre
# L'ordinateur choisit un nombre aléatoire entre 1 et 100
# Le joueur doit trouver ce nombre, à chaque proposition l'ordinateur indique s'il est trop haut ou trop bas.
# L'odinateur indique le nombre de tentatives effectuées
# ###

# Génération d'un nombre aléatoire
# Utilisation de la fonction randint() de la librairie principale de python random

# On importe le module correspondant

# Fonctionnement de randint() à partir de la fonction de base random()
# random() génère un réel dans [0,1[
# from random import random
# on modifie le résultat de random() pour avoir un entier entre 1 et 100
# en utilisant la fonction ceil qui donne l'entier juste inférieur à un réel quelconque
# from math import ceil
# nombre_a_trouver = ceil(random() * 100)
# print(nombre_a_trouver)
# La fonction randint propose directement ce résulat


from random import randint

nombre_a_trouver = randint(1, 100)

print("Trouvez le nombre que j'ai choisi.\nSaisissez votre proposition:")
notFound = True
nbEssai = 0
while (notFound):
    essai = int(input())
    if(essai > nombre_a_trouver):
        nbEssai += 1
        print("Trop haut, essayez à nouveau:")
    elif (essai < nombre_a_trouver):
        nbEssai += 1
        print("Trop bas, essayez à nouveau:")
    else:
        notFound = False
print("Bravo, le nombre à trouver était "+str(nombre_a_trouver)+" vous avez trouvé après "+str(nbEssai)+" essais")




#-----------------
# ###
# Premier petit projet python: Devine le nombre
# L'ordinateur choisit un nombre aléatoire entre 1 et 100
# Le joueur doit trouver ce nombre, à chaque proposition l'ordinateur indique s'il est trop haut ou trop bas.
# L'odinateur indique le nombre de tentatives effectuées
# ###


# Génération d'un nombre aléatoire
# Utilisation de la fonction randint() de la librairie principale de python random


# Fonctionnement de randint() à partir de la fonction de base random()
# random() génère un réel dans [0,1[
# from random import random
# on modifie le résultat de random() pour avoir un entier entre 1 et 100
# en utilisant la fonction ceil qui donne l'entier juste inférieur à un réel quelconque
# from math import ceil
# nombre_a_trouver = 1 + ceil(random() * 100)
# print(nombre_a_trouver)
# La fonction randint propose directement ce résulat
MAXIMUM = 10

nombre_a_trouver = randint(1, MAXIMUM)
# print(nombre_a_trouver)

nombre_propose = 0
essais = 0  # Nombre d'essais


while nombre_propose != nombre_a_trouver:

    # Utilisation d'une boucle while pour s'assurer que l'utilisateur est dans l'intervalle
    # Entrée utilisateur
    nombre_propose = 0
    while not(1 <= nombre_propose <= MAXIMUM):
        nombre_propose = int(input("Proposer un nombre entre 1 et " + str(MAXIMUM) + ": "))

    essais += 1  # On incrémente le nombre d'essais effectués
    # On informe le joueur s'il est trop haut ou trop bas
    if nombre_propose > nombre_a_trouver:
        print("Trop haut")
    elif nombre_propose < nombre_a_trouver:
        print("Trop bas")
    else:
        print("Gagné")
        print("Nombre d'essais effectués: ", essais)