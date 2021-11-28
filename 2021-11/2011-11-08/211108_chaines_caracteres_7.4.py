import sys
import time

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import numpy as np
import ListeUtil as myList
import re
from random import randint

# VARIABLES GLOBALES
VOYELLES = "aàâeéèêëiïîoôuùûy"
DICO = {}

# FUNCTIONS


def load_dico_file(file_name, lang="FR", verbose=False):
    """ Charge le fichier de mot et les ajoute dans la liste de mots
                Args:
                    file_name (String) : nom du fichier
                    verbose (True or False): True pour mode debug
                Returns:
                    None
                """
    if verbose:
        print("Function load_dico_file ", file_name, end="")
    f = open(file_name, "r")
    DICO[lang] = []
    for ligne in f:
        DICO[lang].append(ligne.strip())
    # Fermeture du fichier
    f.close()
    if verbose:
        myList.logEnd("Function load_dico_file " + file_name, verbose)


def sortWordLetter(word, verbose=False):
    """ Tri les lettres d'un mot par ordre alphabétique
            Args:
                word (String) : mot à trier
                verbose (True or False): True pour mode debug
            Returns:
                String : le mot tié
            """
    sorted_list = sorted(list(word))
    sorted_word = ""
    for letter in sorted_list:
        sorted_word += letter
    if verbose:
        print("Function sortWordLetter : ", word," <=> ", sorted_word)
    return sorted_word


def replaceVoyelleBy(word, replace_car="*", verbose=False):
    """ Remplace les voyelles par le caratère reçu
            Args:
                word (String) : mot à trier
                replace_car (String), default *
                verbose (True or False): True pour mode debug
            Returns:
                String : le mot traité
            """
    expression = ""
    for letter in VOYELLES:
        if len(expression)>0:
            expression += "|"
        expression += letter.upper() + "|" + letter.lower()
    reg = re.compile('('+expression+')')
    result_word = reg.sub(replace_car, word)
    if verbose:
        print("Function replaceVoyelleBy : ", word," <=> ", result_word)
    return result_word


def has_word_double_letter(word, verbose=False):
    index = 0
    taille = len(word)
    res = False
    while not res and index < (taille-1):
        if index < taille:
            res = word[index] == word[index+1]
        index += 1
    if verbose:
        print("has_word_double_letter ", word, " = ", res)
    return res


def is_contains_only_voyelle(word, search_car="e", verbose=False):
    """ Remplace les voyelles par le caratère reçu
            Args:
                word (String) : mot à trier
                replace_car (String), default *
                verbose (True or False): True pour mode debug
            Returns:
                String : le mot traité
            """
    # re.search(r'([aeiou])\1', w)]
    res = False
    not_expected_voy = VOYELLES.replace(search_car.lower(), "")
    not_expected_voy += not_expected_voy.upper()
    if not re.search(r'(['+not_expected_voy+'])', word):
        if search_car in word:
            res = True
    if verbose:
        print("Function is_contains_only_voyelle : ", word, " = ", res)
    return res


def is_palindrome(word, verbose=False):
    index = 1
    centre = int(len(word) / 2)
    equ = True
    for l in word:
        l_end = word[(-1 * index)]
        if l == l_end:
            if index >= centre:
                break
            else:
                index += 1
        else:
            equ = False
            break
    return equ


# Proposition prof
def palindromes2(verbose=False):
    palindromes = [mot for mot in DICO["FR"] if mot == mot[::-1]]
    palindromes = []
    for mot in DICO["FR"]:
        if mot == mot[::-1]:
            palindromes.append(mot)


def anacycliques(verbose=False):
    [mot for mot in DICO["FR"] if mot[::-1] in DICO["FR"]  and mot[::-1] != mot ]

    anacycliques = []
    for mot in DICO["FR"]:
        if mot[::-1] in DICO["FR"] and mot[::-1] != mot:
            anacycliques.append(mot)


# -----------------------------------------------------------------------------------------------
#                                                    MAIN
# -----------------------------------------------------------------------------------------------
verbose = False

# Exercice 7.4
# Pour toutes les questions ci-dessous, utilisez le fichier dico.txt.
load_dico_file("211108_dico.txt", "FR", verbose)
nb_words = len(DICO["FR"])

saisi = None
while saisi != "exit":
    saisi = input("\nSaisissez le numéro de l'exercice ou exit. a à h :")
    start_time = time.time()
    moyenne = 0
    nb_match_words = 0
    if saisi == "a":
        print("a) Calculez le pourcentage de mots français où la seule voyelle est le « e » (il peut y en avoir plusieurs dans le mot). Par exemple : exemple, telle, égrener. ")
        for word in DICO["FR"]:
            if is_contains_only_voyelle(word, "e", False):
                nb_match_words += 1
    elif saisi == "b":
        print("b) Calculez le pourcentage de mots français où deux lettres identiques se suivent. Par exemple : femme, panne, créer.")
        for word in DICO["FR"]:
            # vérifie les lettres doublées
            if(has_word_double_letter(word, verbose)):
                nb_match_words += 1
    elif saisi == "c":
        print("c) Affichez les mots de moins de 10 lettres ayant comme lettre centrale un « w ». Le mot doit avoir un nombre impair de lettres. Par exemple : edelweiss, kiwis. ")
        for word in DICO["FR"]:
            taille = len(word)
            if taille % 2:
                centre = taille//2
                if word[centre] == "w" or word[centre] == "W":
                    print(word, ", ", end='')
                    nb_match_words += 1
        print("\n")
    elif saisi == "d":
        print("d) Affichez la liste des mots contenant la suite de lettres « dile ».  Par exemple : crocodile, prédilection.")
        nb_match_words = 0
        for word in DICO["FR"]:
            # vérifie les lettres doublées
            res = re.search(r'(dile)', word)
            if res:
                print(word, ", ", end='')
                nb_match_words += 1
        print("\n")
    elif saisi == "e":
        print("e) Affichez les mots de moins de 5 lettres qui commencent et finissent par la même lettre. Par exemple : aura, croc, dard.")
        for word in DICO["FR"]:
            taille = len(word)
            centre = int(taille / 2)
            # vérifie les lettres doublées
            if len(word) < 5:
                if word[0] == word[-1]:
                    print(word, ", ", end='')
                    nb_match_words += 1
        print("\n")
    elif saisi == "f":
        print("f) Affichez tous les mots palindromes. Par exemple : serres, kayak, ressasser. ")
        for word in DICO["FR"]:
            if is_palindrome(word):
                print(word, ", ", end='')
                nb_match_words += 1
        print("\n")
    elif saisi == "g":
        print("g) Affichez tous les mots anacycliques : un mot lu de droite à gauche donne un autre mot. Par exemple : les – sel, bons – snob.")
        for word in DICO["FR"]:
            # TODO inverser le mot
            # Temps de traitement trop long
            letters = list(word)
            letters.reverse()
            revert_word = "".join(letters)
            # Inverser une chaîne
            # chaine = "abc"
            # chaine[::-1]
            # revert_word = ""
            # for i in range(len(word)):
            #     revert_word += word[-(i+1)]
            if revert_word in DICO["FR"]:
                print(word, "<=> ", revert_word, ", ", end='')
                nb_match_words += 1
        print("\n")
    elif saisi == "h":
        print("h) Affichez tous les mots composés de deux séquences de lettres qui se répètent. Par exemple : papa, chercher.")
        for word in DICO["FR"]:
            taille = len(word)
            centre = int(taille / 2)
            start = word[0:centre]
            end = word[centre:len(word)]
            if start == end:
                print(word, ", ", end="")
                nb_match_words += 1
        print("\n")
    # Affichage des moyennes
    end_time = time.time()
    elapsed = end_time - start_time
    if saisi in "abcdefgh":
        moyenne = (nb_match_words / nb_words) * 100
        print(saisi, ")", nb_match_words, " sur ", nb_words, "soit %.2f" % moyenne, "% en ", elapsed, "s")




