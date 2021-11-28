# dilemme du prisonnier itéré, version duel
from random import choice

choix = ['T', 'C']  # T : trahit, C : coopère


# Dilemme du prisonnier
def gain(lui, moi):
    if lui == 'C' and moi == 'C':
        return 3
    elif lui == 'C' and moi == 'T':
        return 5
    elif lui == 'T' and moi == 'C':
        return 0
    elif lui == 'T' and moi == 'T':
        return 1


# Dilemme de l'ascenseur
def gain2(lui,moi):
    if lui=='C' and moi=='C':
        return 3
    elif lui=='C' and moi=='T':
        return 8 # au lieu de 5
    elif lui=='T' and moi=='C':
        return 0
    elif lui=='T' and moi=='T':
        return 1


# Toujours seul, ne coopère jamais
def toujours_seul(liste_lui, liste_moi):
    return 'T'


# Bonne poire, coopère toujours
def bonne_poire(liste_lui, liste_moi):
    return 'C'


# Aléatoire, joue avec une probabilité égale 'T' ou 'C'
def aleatoire(liste_lui, liste_moi):
    global choix
    return choice(choix)


# Donnant donnant, coopère seulement si l'autre joueur a coopéré au coup précédent.
def donnant_donnant(liste_lui, liste_moi):
    if len(liste_lui) > 0:
        return liste_lui[-1]
    else:  # premier coup
        return 'C'


# Majorité, coopère seulement si l'autre joueur a coopéré en majorité.
def majorite(liste_lui, liste_moi):
    if len(liste_lui) > 0:
        if liste_lui.count('C') > len(liste_lui) // 2:
            return 'C'
        else:
            return 'T'
    else:  # premier coup
        return 'C'


