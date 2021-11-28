from random import randint

# EXO 1
# Lecture du fichier
with open("liste_mots.txt", "r") as fichier:
    liste_mots = fichier.readlines()
    liste_mots =[mot.rstrip() for mot in liste_mots]

# Sélection sans random.choice()
indice_au_hasard = randint(0, len(liste_mots)-1)
print(liste_mots[indice_au_hasard])


# Exercice 7.2
from random import choice

# Récupération des mots
with open("liste_mots.txt", "r") as fichier:
    liste_mots = fichier.readlines()
    liste_mots = [mot.rstrip() for mot in liste_mots]

mot_a_deviner = choice(liste_mots)
# Lettres triées par ordre alphabétique
lettres_triees = ''
for lettre in sorted(mot_a_deviner):
    lettres_triees = lettres_triees + lettre
print(lettres_triees)

mot_propose = ''
while mot_propose.upper() != mot_a_deviner:
    mot_propose = input("Quel est le mot correspondant? ")

print("Gagné")

# EXO 3
texte = "Il était une fois"
texte_etoile = ''
voyelles = 'aeiouyàéèêîûùAEIOUY'
for caractere in texte:
    if caractere in voyelles:
        texte_etoile = texte_etoile + '*'
    else:
        texte_etoile = texte_etoile + caractere

print(texte_etoile)


# Solution Yohann
# list of vowel with special characters
list_vowel = set({'a', 'e', 'i', 'o', 'u', 'à', 'é', 'è', 'ê', 'ë', 'ï', 'ô', 'ù', 'û', 'ü'})

# replace vowel with '*'
def remove_vowel(sentence):
    for vowel in list_vowel:
        sentence = sentence.replace(vowel, '*')
    return sentence


def main():
    sentence = input("Votre phrase : ")
    print(remove_vowel(sentence))