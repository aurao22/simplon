# ---------------------
#  Projet groupes
# ---------------------
# Génération de groupes
# Le but est de générer des groupes de taille déterminée
# de façon aléatoire 
#
# Option avancée:
# en prenant en compte ou pas l'équirépartition H/F
# Si le nombre de groupe n'est pas un diviseur du nombre d'apprenants
# on pourra au choix agrandir certains groupes ou faire un groupe avec moins de personnes
#
#
# Option avancée ++
# Les membres d'un groupes ne peuvent pas se retrouver ensemble
# lors de la prochaine génération de groupes de même taille
import random
import csv

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constantes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NB_GROUPES = 4
STR_HOMME = "homme"
STR_FEMME = "femme"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fonctions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Affiche les clés et les valeurs correspondantes
def display_groups(groupesList):
    print("display_groups --------------------------------------------")
    for cle, valeurs in groupesList.items():
        print("----")
        print(cle)
        print("----")
        for val in range(len(valeurs)):
            print(valeurs[val])
    print("END display_groups -----------------------------------------")


def init_group_list(nb_groups):
    groupes = {}
    for i in range(nb_groups):
        groupeName = "Groupe" + str(i)
        groupes[groupeName] = []
    return groupes


def affect_people_in_groups(groupes, people_list):
    nonSelectionnes = list(people_list)
    nbEleves = len(people_list)
    nbGroup = len(groupes)
    nbElevesParGroupe = nbEleves // nbGroup
    for i in range(nbGroup):
        groupeName = "Groupe" + str(i)
        # print("----")
        # print(groupeName)
        for k in range(nbElevesParGroupe):
            select = random.choice(nonSelectionnes)
            # (select)
            nonSelectionnes.remove(select)
            groupes[groupeName].append(select)

    # Il reste des gens dans la liste
    # print("-----------------------------------------")
    rest = len(nonSelectionnes)
    # print(rest)
    for i in range(rest):
        idx = random.randint(0, len(nonSelectionnes) - 1)
        groupeName = "Groupe" + str(i)
        groupes[groupeName].append(nonSelectionnes.pop(idx))
        # print(groupeName, groupes[groupeName])
    return groupes


def load_people_file():
    f = open(r"C:\Users\User\Documents\SIMPLON\workspace\jour1\211019_groupes-Eleves.csv")
    myReader = csv.reader(f)

    hommes = []
    femmes = []
    # Contenu du fichier csv
    # Prénom,Nom,genre,adresse mail,Pseudo Discord
    for row in myReader:
        if STR_HOMME in row:
            hommes.append(row)
        else:
            femmes.append(row)
    print("-----------------------------------------")
    print(femmes)
    print(hommes)
    print("-----------------------------------------")
    return femmes, hommes


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
femmes, hommes = load_people_file()
groupes = init_group_list(NB_GROUPES)

affect_people_in_groups(groupes, hommes)
affect_people_in_groups(groupes, femmes)
display_groups(groupes)

# TODO - Sauvegarder le groupe de chacun




