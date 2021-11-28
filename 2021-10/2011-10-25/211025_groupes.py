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
import os

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
    print("Initialisation des groupes...")
    groupes = {}
    for i in range(nb_groups):
        groupeName = "Groupe" + str(i)
        groupes[groupeName] = []
    print("Initialisation des groupes... TERMINE")
    return groupes


def affect_people_in_groups(groupes, people_list):
    print("Affectation des personnes par groupe...")
    nonSelectionnes = list(people_list)
    nbEleves = len(people_list)
    nbGroup = len(groupes)
    nbElevesParGroupe = nbEleves // nbGroup
    for i in range(nbGroup):
        groupeName = "Groupe" + str(i)
        # print("----")
        # print(groupeName)
        k = 0
        while( k < nbElevesParGroupe):
            select = random.choice(nonSelectionnes)
            # vérifier que la personne n'était pas déjà dans ce groupe
            previousGroup = select[len(select) - 1]
            # S'il était dans le groupe, on passe à l'itération suivante
            if(previousGroup != groupeName):
                # (select)
                nonSelectionnes.remove(select)
                select[len(select) - 1] = groupeName
                groupes[groupeName].append(select)
                # Il faut remplacer l'ancien nom de groupe avec le nouveau
                k += 1

    # Il reste des gens dans la liste
    # print("-----------------------------------------")
    rest = len(nonSelectionnes)
    # print(rest)
    i = rest
    while(i > 0):
        idx = random.randint(0, len(nonSelectionnes) - 1)
        groupeName = "Groupe" + str(i)
        select = nonSelectionnes[idx]
        # print(groupeName, groupes[groupeName])
        previousGroup = select[len(select) - 1]
        # S'il était dans le groupe, on passe au tirage aléatoire suivant
        if (previousGroup != groupeName):
            del nonSelectionnes[idx]
            select[len(select) - 1] = groupeName
            groupes[groupeName].append(select)
            # Il faut remplacer l'ancien nom de groupe avec le nouveau
            i -= 1
    print("Affectation des personnes par groupe... TERMINE")
    return groupes


def load_people_file():
    print("Chargement du fichier...")
    f = open(r"211025_groupes-Eleves.csv")
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
    print("Chargement du fichier... TERMINE")
    print("-----------------------------------------")
    print(femmes)
    print(hommes)
    print("-----------------------------------------")
    return femmes, hommes


# Ecrit le fichier pour sauvegarder les groupe
def writeFile(fileName, data):
    # TODO faut-il créer le fichier avant ?
    # TODO faut-il vider le fichier avant ?
    if os.path.exists(fileName):
        print("Suppression du fichier", fileName)
        os.remove(fileName)
        print("Suppression du fichier", fileName, "... TERMINE")
    else:
        print("Le fichier n'existe pas")
    print("création du fichier", fileName,"et écriture...")
    f = open(fileName, 'w+')
    for key, line in data.items():
        for s in line:
            str = ""
            for st in s:
                str += st + ","
            # On écrit toute la ligne sauf la dernière virgule
            f.write(str[:-1])
            f.write('\n')
    f.close()
    print("création du fichier", fileName,"et écriture...TERMINE")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
femmes, hommes = load_people_file()
groupes = init_group_list(NB_GROUPES)

affect_people_in_groups(groupes, hommes)
affect_people_in_groups(groupes, femmes)
display_groups(groupes)

# Sauvegarder le fichier avec le nouveau groupe de chacun
writeFile("211025_groupes-Eleves_result.csv", groupes)






