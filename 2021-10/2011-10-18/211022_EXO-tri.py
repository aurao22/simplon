import numpy as np
import matplotlib.pyplot as plt
from random import sample

# Exercice 1 : Tri par Insertion

# Ex 1.1
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


# Ex 1.2
#On considère la liste L = [8 , 5 , 3 , 9 , 2]. Écrire le programme principal permettant de trier la liste.
L = [8 , 5 , 3 , 9 , 2]
print(L)
res = tri_insertion(L.copy())
print(res)

# Exercice 2
def createFile(fileName):
    # Ce bout de code permet d'écrire le fichier
    with open(fileName, 'w') as f:
        for i in range(0, 20):
            f.write(f'{i}: {i ** 2} \n')
        f.close()


# Écrivez ici le code pour lire le fichier et enregistrer chaque lignes dans une liste.
def loadFile(fileName):
    # Lire le fichier
    f = open(fileName, 'r')
    lines = f.readlines()
    # fermez le fichier après avoir lu les lignes
    f.close()

    linesList = []
    # Itérer sur les lignes
    for line in lines:
        #read().splitlines()
        line = line.strip().split(sep=': ')
        #line = line.splitlines()
        linesList.append(line)
    return linesList


fileName = 'fichier.txt'

createFile(fileName)
linesList =loadFile(fileName)
print("--------------------------")
# items = [int(x) for x in items]
[print(x) for x in linesList]

# Exercice 3
# Créez une fonction "graphique" qui permet de tracer sur une seule et meme figure une série de graphiques issue d'un dictionnaire contenant plusieurs datasets :

colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
#ligneStyle = ["-", "--", ":", "-."]
ligneStyle = ['dashed', 'dashdot', 'dotted','solid', 'None']
#marqeurStyle = ["o", "x", "X", "--", "*", ".", ",", "V", "^", "<", ">", "1", "2", "3", "4", "s", "p", "h", "H", "+", "P", "d", "D", "|", "_"]
marqeurStyle = ["o", "x", "X", "*", ".",  "+", "_"]


def graphique(dataset):

    n = len(dataset)
    combine = []
    for key, i in zip(dataset.keys(), range(1, n+1)):
        # TODO : Ajouter la série et la légende sur le graphe
        found = False
        while(not found):
            c = sample(colors, 1)
            ls = sample(ligneStyle, 1)
            ms = sample(marqeurStyle, 1)
            style = c[0] + ms[0] + ls[0]
            if style not in combine:
                combine.append(style)
                found = True
        # Définition du sous graphe
        plt.subplot(n, 1, i)
        # Paramétrage du sous graphique
        #plt.style.use('dark_background')
        plt.grid()
        plt.title(key)
        plt.plot(dataset[key],color=c[0], marker=ms[0], linestyle=ls[0])

    #plt.figure(figsize=(12, 20))
    #plt.style.use('dark_background')
    plt.show()


# Voici le dataset utilisé
dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}
graphique(dataset)
