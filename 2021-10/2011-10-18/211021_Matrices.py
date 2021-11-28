import numpy as np

def initialisation_bis(m, n):
    # m : nombre de lignes
    # n : nombre de colonnes
    # retourne une matrice aléatoire (m, n+1)
    # avec une colonne biais (remplie de "1") tout a droite
    X = np.random.randn(m, n)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    return X

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

def exo3():
    print("---------------------------- EXERCICE 3 -----------------------------------")
    matrice = initialisation(5, 4)
    printMatrice(matrice)
    print("---------------------------------------------------------------------------")

#Créer deux matrices 𝑅=(142536) et 𝑆=Les afficher avec Python.
def exo4():
    print("---------------------------- EXERCICE 4 -----------------------------------")
    # affectation des valeurs
    r = np.array([[1,2,3],[4,5,6]])
    s = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print("R=", r)
    print("S=", s)
    #n = entier naturel non nul, renvoie  zéro dans tous les autres cas
    print("R, donc attendu 0, obtenu :", testAvanceAuto(r))
    print("S, donc attendu 9, obtenu :", testAvanceAuto(s))

    # Déterminer les valeurs propres et les vecteurs propres de la matrice 𝐴=(56−3−4) en utilisant la fonction eig
    a = np.array([[5, -3], [6, -4]])
    res = np.linalg.eig(a)
    print("valeurs propres de a:", res)
    print("---------------------------- ")
    print("valeurs propres de a:", np.linalg.eigvals(a))


#Renvoyant la valeur n si M est une matrice carrée d’ordre n (entier naturel non nul),
# et zéro dans tous les autres cas.
def testAvanceManuel(matrice):
    # Vérifie que la matrice a autant de ligne que de colonne
    nbDim = matrice.shape
    res = 0
    if(nbDim[0]==nbDim[1]):
        # res = nbDim[0]
        res = matrice.size
        for ligne in range(nbDim[0]):
            # Initialisation du tableau représentant la ligne
            lineTab = matrice[ligne]
            for col in range(nbDim[1]):
                nb = lineTab[col]
                # Vérifier que les valeurs dans la matrice sont des entiers naturel non nul
                if(not np.issubdtype(nb, np.integer) or nb < 0):
                     res = 0
                     break
    return res

def testAvanceAuto(matrice):
    # Vérifie que la matrice a autant de ligne que de colonne
    nbDim = matrice.shape
    res = 0
    if(nbDim[0]==nbDim[1]):
        res = matrice.size
        # Vérifier que les valeurs dans la matrice sont des entiers naturel non nul
        if(not np.issubdtype(matrice.dtype, np.integer)):
             res = 0
    return res

def test(matrice):
    # Vérifie que la matrice a autant de ligne que de colonne
    nbDim = matrice.shape
    res = 0
    if(nbDim[0]==nbDim[1]):
        res = matrice.size
    return res

exo4()

def exo5():
    print("---------------------------- EXERCICE 5 -----------------------------------")
    L1 = []
    for i in range(2, 7, 2):
        L1.append(i)
    print('L1 = ', L1)
    print("L1[:-1]=",L1[:-1])
    T1 = np.arange(2, 7, 2)
    print('T1 = ', T1)
    print("T1[1:3]=",T1[1:3])
    x1 = np.linspace(0, np.pi, 5)
    print('x1 = ', x1)
    print('x1[1:3] = ', x1[1:3])


#exo5()
print("---------------------------------------------------------------------------")
