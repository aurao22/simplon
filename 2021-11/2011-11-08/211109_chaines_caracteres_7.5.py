import sys
import time

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import ListeUtil as myList


# VARIABLES GLOBALES
VOYELLES = "aàâeéèêëiïîoôuùûy"
VOYELLES = VOYELLES + VOYELLES.upper()
LETTRES_COURANTES = "EAISTNRULODMP"
LETTRES_COURANTES = LETTRES_COURANTES.lower() + LETTRES_COURANTES
LETTRES_RARES = "CVQGBFJHZXYKW"
LETTRES_RARES = LETTRES_RARES.lower() + LETTRES_RARES

DICO = {}

# FUNCTIONS

def load_dico_file(file_name, lang="FR", verbose=False):
    """
    Charge le fichier de mot et les ajoute dans la liste de mots
    :param file_name (String) : nom du fichier
    :param verbose (True or False): True pour mode debug
    :return: None
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


def is_rare_word(word, verbose=True):
    """
    Affichez la liste des mots n'ayant que des voyelles et des lettres rares.
    :param word:
    :param verbose: (True or False): True pour mode debug
    :return: True s'il n'y a que des lettres rares et des voyelles dans le mot
    """
    res = True
    for letter in word:
        if letter in LETTRES_RARES:
            continue
        elif letter in VOYELLES:
            continue
        else:
            res = False
            break;
    return res


def is_courant_word(word, verbose=True):
    """
    Affichez la liste des mots de plus de 15 lettres n'ayant que des lettres courantes.
    :param word:
    :param verbose: (True or False): True pour mode debug
    :return:
    """
    res = True
    for letter in word:
        if letter in LETTRES_COURANTES:
            continue
        else:
            res = False
            break;
    return res


def is_word_contains_one_letter_or_more(word, letters, verbose=True):
    """
    Affichez la liste des mots n'ayant aucune des lettres E A I S T N R U
    :param word:
    :param verbose:
    :return:
    """
    letters = letters.lower() + letters.upper()
    res = False
    for letter in word:
        if letter in letters:
            res = True
            break
    return res


def get_nb_voyelles(word, verbose=False):
    """
    Compte le nombre de voyelles dans le mot
    :param word (String) : mot à trier
    :param verbose (True or False): True pour mode debug
    :return: int : le nombre de voyelles
    """
    nb_voyelle = 0
    for letter in word:
        if(letter in VOYELLES):
            nb_voyelle += 1
    if verbose:
        print("Function get_nb_voyelles : ", word," <=> ", nb_voyelle)
    return nb_voyelle


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
    saisi = input("\nSaisissez le numéro de l'exercice ou exit. a à e :")
    start_time = time.time()
    moyenne = 0
    nb_match_words = 0
    if saisi == "a":
        print("a) Affichez la liste des mots n'ayant que des voyelles et des lettres rares")
        for word in DICO["FR"]:
            if is_rare_word(word, verbose):
                print(word, ", ", end="")
                nb_match_words += 1
        print("\n")
    elif saisi == "b":
        print("b) Affichez la liste des mots de plus de 15 lettres n'ayant que des lettres courantes. ")
        for word in DICO["FR"]:
            if len(word)>15 and is_courant_word(word, verbose):
                print(word, ", ", end="")
                nb_match_words += 1
        print("\n")
    elif saisi == "c":
        print("c) Affichez la liste des mots n'ayant aucune des lettres E A I S T N R U ")
        for word in DICO["FR"]:
            if not is_word_contains_one_letter_or_more(word, "EAISTNRU", verbose):
                print(word, ", ", end="")
                nb_match_words += 1
        print("\n")
    elif saisi == "d":
        print("d) Affichez la liste des mots ayant exactement deux voyelles et plus de 9 caractères (tiret compris). Par exemple : transports, check-list.")
        for word in DICO["FR"]:
            if len(word)>8:
                nb_voy = get_nb_voyelles(word, verbose)
                if nb_voy == 2:
                    print(word, ", ", end="")
                    nb_match_words += 1
        print("\n")
    elif saisi == "e":
        print("e) Affichez la liste des mots commençant par au moins 4 consonnes consécutives (tiret non autorisé)")
        for word in DICO["FR"]:
            found = False
            if len(word) > 3:
                for i in range(4):
                    if word[i] in VOYELLES or word[i] == "-":
                        found = True
                        break
                if not found:
                    print(word, ", ", end="")
                    nb_match_words += 1
        print("\n")

    # Affichage des moyennes
    end_time = time.time()
    elapsed = end_time - start_time
    if saisi in "abcde":
        moyenne = (nb_match_words / nb_words) * 100
        print(saisi, ")", nb_match_words, " sur ", nb_words, "soit %.2f" % moyenne, "% en ", '{0:.3f}'.format(elapsed), "s")
