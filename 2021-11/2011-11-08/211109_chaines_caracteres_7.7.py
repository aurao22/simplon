import sys
import time

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import ListeUtil as myList

# VARIABLES GLOBALES
VOYELLES = "aàâeéèêëiïîoôuùûy"
VOYELLES = VOYELLES + VOYELLES.upper()
DICO_WORD = {}
DICO_DICO = {}


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
    DICO_WORD[lang] = []
    cle = []
    for ligne in f:
        word = ligne.strip()
        DICO_WORD[lang].append(word)
        taille = len(word)
        if taille not in cle:
            DICO_DICO[taille] = []
            cle.append(taille)
        word = word.replace("-", "")
        word = word.replace("'", "")
        DICO_DICO[taille].append(word)

    # Fermeture du fichier
    f.close()
    if verbose:
        myList.logEnd("Function load_dico_file " + file_name, verbose)


# -----------------------------------------------------------------------------------------------
#                                                    MAIN
# -----------------------------------------------------------------------------------------------
verbose = False

load_dico_file("211108_dico.txt", "FR", verbose)
nb_words = len(DICO_WORD["FR"])

saisi = None
while saisi != "exit":
    saisi = input("\nEntrez votre tirage (10 lettres) or exit :")
    start_time = time.time()
    moyenne = 0
    nb_match_words = 0
    if saisi != "exit":
        for word in DICO_WORD["FR"]:
                print(word, ", ", end="")
                nb_match_words += 1
        print("\n")
        end_time = time.time()
        elapsed = end_time - start_time
           # Affichage des moyennes
        moyenne = (nb_match_words / nb_words) * 100
        print(nb_match_words, " sur ", nb_words, "soit %.2f" % moyenne, "% en ", elapsed, "s")

