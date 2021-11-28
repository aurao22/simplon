import inspect

def logEnd(start, repeatStart=False):
    strEnd = "END"
    taille = 100 - len(start)
    for s in range(taille):
        strEnd = "." + strEnd
    if repeatStart:
        print(start, strEnd)
    else:
        print(strEnd)


# Écrire une fonction tri_insertion qui admet comme argument uneliste L et permet de la trier par ordre croissant en utilisant la méthode du tri par insertion.
def tri_insertion(list, idx=1):
    print(idx)
    if list is not None:
        if idx < len(list):
            current = list.pop(idx)
            for i in range (0, idx):
                if list[i]<current:
                    continue
                else:
                    list.insert(i, current)
                    return tri_insertion(list, idx+1)
            # Traitement du cas où il s'agit du dernier élément à insérer
            list.insert(i+1, current)
            return tri_insertion(list, idx + 1)
    return list


def initialisation(m, n):
    # m : nombre de lignes
    # n : nombre de colonnes
    # retourne une matrice aléatoire (m, n+1)
    # avec une colonne biais (remplie de "1") tout a droite

    #Initialisation de la matrice avec des valeurs aléatoires
    matrice = np.zeros((m, n+1))
    print('The value is :', matrice)
    # if we check the matrix dimensions
    # using shape:
    print('The shape of matrix is :', matrice.shape)
    # by default the matrix type is float64
    print('The type of matrix is :', matrice.dtype)
    # Ajout de la colonne de biais

    for ligne in range(m):
        # Initialisation du tableau représentant la ligne
        lineTab = matrice[ligne]
        for col in range(n + 1):
            # On traite le cas du biais
            nb = 1
            if (col < n):
                nb = np.random.randint(-100, 100)
            lineTab[col] = nb
        # Ajout de la ligne à la matrice générale
        #matrice[ligne] = lineTab
    print('The value is :', matrice)
    return matrice


def printMatrice(matrice):
    # nombre de lignes et col, shape renvoie un tuple
    nbDim = matrice.shape
    nbLigne = nbDim[0]
    nbCol = nbDim[1]
    print("nbDim:",nbDim,"nbLigne:",nbLigne,"x nbCol:",nbCol)
    print(matrice)