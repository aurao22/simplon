import sys
import time

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')

"""
L'argot javanais, apparu en France dans la dernière moitié du 19ème siècle, consiste à intercaler
dans les mots la syllabe « av » entre une consonne et une voyelle. Nous n'utiliserons ici que les
règles simplifiées (*) :
"""
# VARIABLES GLOBALES
VOYELLES = "aàâeéèêëiïîoôuùûy"
VOYELLES = VOYELLES + VOYELLES.upper()


# FUNCTIONS
def encode_javanais(sentence, verbose=False):
    """
    Encode la phrase en javanais
    :param sentence: la phrase à encoder
    :param verbose: (True or False): True pour mode debug
    :return: La phrase encodée
    """
    encoded_sentence = ""
    javanais_car = "av"
    previous_consonne = True
    for letter in sentence:
        if letter in VOYELLES :
            if previous_consonne:
                encoded_sentence += javanais_car
            previous_consonne = False
        else:
            previous_consonne = True

        encoded_sentence += letter
    if verbose:
        print("Function encode_javanais : \n", sentence, "\n=> ", encoded_sentence)
    return encoded_sentence


# -----------------------------------------------------------------------------------------------
#                                                    MAIN
# -----------------------------------------------------------------------------------------------
verbose = False

"""
L'argot javanais, apparu en France dans la dernière moitié du 19ème siècle, consiste à intercaler
dans les mots la syllabe « av » entre une consonne et une voyelle. Nous n'utiliserons ici que les
règles simplifiées (*) :
• On rajoute « av » après chaque consonne (ou groupe de consonnes comme par exemple ch,
cl, ph, tr,…) d'un mot.
• Si le mot commence par une voyelle, on ajoute « av » devant cette voyelle.
• On ne rajoute jamais « av » après la consonne finale d'un mot.
"""

verbose = True
saisi = "Je vais acheter des allumettes au supermarché."

while saisi != "exit":
    saisi = input("\nSaisissez une phrase, ex :" + saisi)
    start_time = time.time()

    if saisi != "exit":
        encoded = encode_javanais(saisi)
        print(encoded)
        end_time = time.time()
        elapsed = end_time - start_time
        print("in",  elapsed, "s")
