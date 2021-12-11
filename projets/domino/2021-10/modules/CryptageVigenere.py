

def cryptLettre(lettre, cle, idx) :
    nbCar = getKeyLetterPosition(cle, idx)
    nbCarLettre = getLetterPosition(lettre)
    nextIdx = nbCar + nbCarLettre
    if nextIdx > 25:
        nbCar = nbCar - (26 - nbCarLettre)
        if (lettre.isupper()):
            lettre = "A"
        else:
            lettre = "a"
    nextLettre = chr(ord(lettre) + nbCar)
    return nextLettre


def unCryptLettre(lettre, cle, idx):
    nbCar = getKeyLetterPosition(cle, idx)
    nbCarLettre = getLetterPosition(lettre)
    nextIdx = nbCarLettre - (nbCar)

    if(nextIdx <  0):
        nbCar = -nextIdx -1
        if (lettre.isupper()):
            lettre = "Z"
        else:
            lettre = "z"
    nextLettre = chr(ord(lettre)-nbCar)
    return nextLettre


def getNextCleIdx(cle, currentIdx):
    cleIdx = currentIdx
    if cleIdx < (len(cle)-1):
        cleIdx += 1
    else:
        cleIdx = 0
    return cleIdx


def getKeyLetterPosition(cle, cleIdx):
    return getLetterPosition(cle[cleIdx])


def getLetterPosition(letter):
    letterAlphabetPosition = ord(letter.lower())-ord("a")
    return letterAlphabetPosition


def code_vigenere(message, cle):
    messageCrypte = ""
    cleIdx = 0

    for i in range(0, len(message)):
        car = message[i]
        newCar = car
        if car.isalpha():
            newCar = cryptLettre(newCar, cle, cleIdx)
            cleIdx = getNextCleIdx(cle, cleIdx)
        messageCrypte += newCar
    return messageCrypte


def decode_vigenere(message_code, cle):
    messageDeCrypte = ""
    cleIdx = 0

    for i in range(0, len(message_code)):
        car = message_code[i]
        newCar = car
        if car.isalpha():
            newCar = unCryptLettre(car, cle, cleIdx)
            cleIdx = getNextCleIdx(cle, cleIdx)
        messageDeCrypte += newCar
    return messageDeCrypte

#
# def cryptDecryptMessage(message, cle):
#     print('message : ', message)
#     print('clé  : ', cle)
#
#     message_code = code_vigenere(message, cle)
#     print('message codé : ', message_code)
#
#     message_initial = decode_vigenere(message_code, cle)
#     print('message initial  : ', message_initial)



# cryptDecryptMessage('abcz', 'ab')
# message = "Parcours delivrant un titre a finalite professionnelle et une certification reconnus par l’Etat, ainsi qu’une certification Microsoft Azure."
# cryptDecryptMessage(message, "Simplon")