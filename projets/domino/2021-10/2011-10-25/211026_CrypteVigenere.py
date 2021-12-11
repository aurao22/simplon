import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import CryptageVigenere as cv


def cryptDecryptMessage(message, cle):
    print('message : ', message)
    print('clé  : ', cle)

    message_code = cv.code_vigenere(message, cle)
    print('message codé : ', message_code)

    message_initial = cv.decode_vigenere(message_code, cle)
    print('message initial  : ', message_initial)

# cryptDecryptMessage('abcz', 'ab')

message = "Parcours delivrant un titre a finalite professionnelle et une certification reconnus par l’Etat, ainsi qu’une certification Microsoft Azure."

cryptDecryptMessage(message, "Simplon")