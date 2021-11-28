import matplotlib.pyplot as plt
import math
import numpy as np

def f1(x): # définition de la fonction 
    if x>=0 and x<1:
            return x 
    elif x >= 1 and x <=2:
        return 1
    else:
        return 0
def f2(x): # définition de la fonction f2return 
    math.sin(x)+0.1
    
xmin=0 # valeur minimale de x
xmax=2 # valeur maximale de x
pas=0.05
N=int(((xmax-xmin)/pas)+1) # pas=(xmax-xmin)/(N-1)# le nombre de points doit être un entier# N=nombre de points et N-1=nombre d'intervalles
x=[ ] # initialisation de la liste x
y1=[ ] # initialisation de la liste y1
y2=[ ] # initialisation de la liste y2
for i in range(N):
    x.append(i*pas) #ajout de l'élément x[i]
    y1.append(f1(x[i])) # ajout de l'élément f1(x[i])
    y2.append(f2(x[i])) # ajout de l'élément f2(x[i])
len(x)
len(np.arange(0,2,0.05))


def F1(x): # définition de la fonction F1
    if x>=0 and x<1:
        return x
    elif x >= 1 and x <= 2:
        return 1
    else:
        return 0
    
    
def F2(x): # définition de la fonction F2
    return np.sin(x)+0.1


N=21 # nombre de points
X=np.linspace(xmin,xmax,N) # valeurs entre xmin et xmax compris
F1_vect=np.vectorize(F1)# transforme la fonction scalaire en une fonction# pouvant être utilisée avec des tableaux numpy
Y1=F1_vect(X) # applique F1 à chaque élément de X# temps de calcul moins long qu'avec des boucles
Y2=F2(X) # fonction déjà vectorisée avec np.sin