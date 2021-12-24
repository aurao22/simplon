from scipy import misc
import matplotlib.pyplot as plt


def drawPicture(face):
    plt.imshow(face, cmap=plt.cm.gray)
    plt.show()


# Écrire un programme Python permettant d’obtenir le nombre de lignes et le nombre de colonnes
# reduceType = center (0), bottom (1), top(2), left(3), right(4)
def reducePicture(picture, divLine, divCol, reduceType):
    miPicture = picture.copy()
    nbDim = miPicture.shape
    nbLine = nbDim[0]
    nbCol = nbDim[1]

    # Sur l'image, effectuer un slicing pour ne garder que la moitié de l'image (en son centre)
    # Traitement des cas par défaut, au centre
    startLine = 0
    endLine = nbLine
    startCol = 0
    endCol = nbCol

    if reduceType == 1: # Bottom
            startLine = nbLine-(int((nbLine / (divLine/2))))
    elif reduceType == 2:   # 'top':
            endLine = startLine + (int((nbLine / divLine)))
    elif reduceType == 3:   # 'left':
            endCol = startCol + (int((nbCol / divCol)))
    elif reduceType == 4:   #'right':
            startCol = nbCol - (int((nbCol / divCol)))
    # Sur l'image, effectuer un slicing pour ne garder que la moitié de l'image (en son centre)
    # Traitement des cas par défaut, au centre
    else:
        startLine = int((nbLine / divLine))
        endLine = nbLine - startLine
        startCol = int(nbCol / divCol)
        endCol = nbCol - startCol

    miPicture = miPicture[startLine:endLine,startCol:endCol]
    return miPicture


# et remplacer tous les pixels > 150 par des pixels = 255
def cleanColor(pict, colorToChange, newColor):
    newPict = pict.copy()
    newPict[newPict > colorToChange] = newColor
    return newPict


def limitGray(pict):
    pict[pict <= 80] = 60
    pict[(pict > 80) & (pict < 150)] = 120
    pict[pict >= 150] = 220
    return pict


def compressPict(pict, nb):
    pict = pict[::nb,::nb]
    return pict



# Écrire un programme Python permettant d’obtenir le nombre de lignes et le nombre de colonnes
# reduceType = center, bottom, top, left, right
def reducePictureMatch(picture, divLine, divCol, reduceType):
    miPicture = picture.copy()
    nbDim = miPicture.shape
    nbLine = nbDim[0]
    nbCol = nbDim[1]

    reduceType = 0   # "center"
    reduceTypeKey = "reduceType"
    for key, value in kwargs.items():
        print("%s = %s" % (key, value))
        if reduceTypeKey == key:
            reduceType = int(value)

    # Sur l'image, effectuer un slicing pour ne garder que la moitié de l'image (en son centre)
    # Traitement des cas par défaut, au centre
    startLine = int((nbLine / divLine))
    endLine = nbLine - startLine
    startCol = int(nbCol / divCol)
    endCol = nbCol - startCol

    # match reduceType:
    #     case 1:   # "bottom":
    #         startLine = nbLine-(int((nbLine / divLine)))
    #         endLine = nbLine
    #     case 2:   # 'top':
    #         startLine = 0
    #         endLine = startLine + (int((nbLine / divLine)))
    #     case 3:   #'left':
    #         startCol = 0
    #         endCol = startCol + (int((nbCol / divCol)))
    #     case 4:   #'right':
    #         startCol = nbCol - (int((nbCol / divCol)))
    #         endCol = nbCol

    miPicture = miPicture[startLine:endLine,startCol:endCol]
    return miPicture