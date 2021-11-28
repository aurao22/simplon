# Le jeu du pendu
 
from random import choice
 
fichier = open("liste_mots.txt", "r")
liste_mots = fichier.readlines()    # met tous les mots du fichiers dans une liste
mot = choice(liste_mots)            # prend au hasard un mot dans la liste
mot = mot.rstrip()                  # supprime le caractère "saut à la ligne"
fichier.close()

mot_devine = "-" * len(mot)
print(mot_devine)
nbr_essais = 0

while mot_devine != mot:
    lettre = input("Entrez une lettre ou '?' pour abandonner : ")
    lettre = lettre[0]  # évite des erreurs si un mot est entré au lieu d'une lettre
    if lettre == '?':
        print('Le mot était',mot)
        break
    lettre = lettre.upper()
    for i in range(len(mot)):
        if lettre == mot[i]:
            mot_devine = mot_devine[:i] + lettre + mot_devine[i+1:]
    print(mot_devine)
    nbr_essais += 1
    
if mot == mot_devine:
    print('Bravo ! Le mot',mot,'a été trouvé en',nbr_essais, 'coups')

