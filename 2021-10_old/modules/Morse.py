import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')

import inspect
import ListeUtil as myList
import winsound
import time

MORSE_CODE_DICT = { 'A':'.-', 'B':'-...',
                    'C':'-.-.', 'D':'-..', 'E':'.',
                    'F':'..-.', 'G':'--.', 'H':'....',
                    'I':'..', 'J':'.---', 'K':'-.-',
                    'L':'.-..', 'M':'--', 'N':'-.',
                    'O':'---', 'P':'.--.', 'Q':'--.-',
                    'R':'.-.', 'S':'...', 'T':'-',
                    'U':'..-', 'V':'...-', 'W':'.--',
                    'X':'-..-', 'Y':'-.--', 'Z':'--..',
                    '1':'.----', '2':'..---', '3':'...--',
                    '4':'....-', '5':'.....', '6':'-....',
                    '7':'--...', '8':'---..', '9':'----.',
                    '0':'-----', ',':'--..--', '.':'.-.-.-',
                    '?':'..--..', '/':'-..-.', '-':'-....-',
                    '(':'-.--.', ')':'-.--.-',
                    'erreur':'........',
                    'start':'-.-.-',
                    'end':'.-.-.'}

# The frequency parameter specifies frequency, in hertz, of the sound, and must be in the range frequency must be in 37 thru 32767
# Oreille humaine entre 20 et 20 000 Hz
FREQUENCIES = {".": 1500, "-": 1500}
# The duration parameter specifies the number of milliseconds the sound should last.
DURATIONS = {".": 800, "-": 1200}


def code_text(message, verbose=False):
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName, end='')
    messageCrypte = []

    for car in message.upper():
        newCar = MORSE_CODE_DICT.get(car, car)
        messageCrypte.append(newCar)

    myList.logEnd("Function " + functionName, verbose)
    return messageCrypte


def decode_text(message, verbose=False):
    """ Décode le message reçu
            Args:
                message ([string]): message encodé en morse
                verbose (True or False): True pour mode debug
            Returns:
                String : Message décrypté
            """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    if verbose:
        print("Function", functionName)
    else:
        print("Function", functionName, end='')
    message_clear = ""

    alpha_morse_dict = {}

    for k, v in MORSE_CODE_DICT.items():
        alpha_morse_dict[v] = k
        if verbose:
            print(v, ":", k)

    for car in message:
        newCar = alpha_morse_dict.get(car, car)
        message_clear += newCar
        if verbose and car == newCar:
            print(car, ":", newCar)

    myList.logEnd("Function " + functionName, verbose)
    return message_clear


def playMorse(messageCrypte, verbose=False):
    """ Joue le morse avec les sons de la machine
                Args:
                    messageCrypte ([string]): message encodé en morse
                    verbose (True or False): True pour mode debug
                Returns:
                    Nothing
                """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    print("Function", functionName)
    for car in messageCrypte:
        try:
            playMorseCar(car, verbose)
            # Séparation des mots
            time.sleep(1)
            if car == " ":
                print("| ", end='')
        except RuntimeError:
            # On prévient qu'il y a une erreur, mais on continue la transmission
            playMorseCar(MORSE_CODE_DICT["erreur"])
    myList.logEnd("\nFunction " + functionName, True)


def playMorseCar(carCrypte, verbose=False):
    """ Joue le morse avec les sons de la machine
                Args:
                    messageCrypte ([string]): message encodé en morse
                    verbose (True or False): True pour mode debug
                Returns:
                    Nothing
                """
    frame = inspect.currentframe()
    functionName = inspect.getframeinfo(frame).function
    if verbose:
        print("Function", functionName)
    for car in carCrypte:
        if FREQUENCIES.get(car, False) and DURATIONS.get(car, False):
            winsound.Beep(FREQUENCIES[car], DURATIONS[car])
            print(car, end='')
        elif verbose:
            print("Caractère non morsable:", car)
    # Séparation des caractères
    time.sleep(0.5)
    print(" ", end="")
    if verbose:
        myList.logEnd("Function " + functionName, verbose)


# ----------------------------------------------------------------------
test = True
verbose = False

if test:
    # Début du test du programme de test
    message = "code morse"
    expected = ['-.-.', '---', '-..', '.', ' ', '--', '---', '.-.', '...', '.']
    # Encodage du message
    messageCrypte = code_text(message, verbose)

    print("messageCrypte", messageCrypte)
    print("expected     ", expected)

    if messageCrypte == expected:
        if verbose: print("BRAVO - Encodage réussi")
        # Décodage du message
        messageClear = decode_text(messageCrypte, verbose)
        if verbose:
            print(messageClear.lower())
            if messageClear.lower() == message:
                print("BRAVO - Décode réussi")
            else:
                print("FAIL - Echec du décodage")

        playMorse(messageCrypte, verbose)
    else:
        print("FAIL - Echec de l'encodage")
    print("END")
