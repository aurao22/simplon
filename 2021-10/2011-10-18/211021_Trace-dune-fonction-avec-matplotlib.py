import math
import matplotlib.pyplot as plt
import numpy as np


# 𝑓2(𝑥)=
# 𝑥 pour0≤𝑥<1
# 1 pour1≤𝑥≤2
# 𝑓1 et 𝑓2 déﬁnies sur [0,2]
def f1(x):
    if 0 <= x < 1:
        return x
    elif 1 <= x <= 2:
        return 1
    else:
        return None


# 𝑓2 définies sur [0,2]
# 𝑓2(𝑥)=𝑠𝑖𝑛(𝑥)+0,1
def f2(x):
    return math.sin(x) + 0.1


def generateMatriceFXManuel(fonc, input_tab):
    nbDim = len(input_tab)
    res = []
    for i in range(nbDim):
        res.append(fonc(input_tab[i]))
    return res


def generateMath(min, max, pas):
    tab = []
    i = min
    while i < (max+pas):
        tab.append(i)
        i += pas
    return tab


#Tracer les courbes représentatives des deux fonctions sur l’intervalle [0,2] avec un pas de 0,05.
def draw_f1_and_f2():
    #[0,2] avec un pas de 0,05
    #tab = np.arange(0, 2, 0.05)
    tab = generateMath(0, 2, 0.05)
    print(tab)
    f1_val = generateMatriceFXManuel(f1, tab)
    f2_val = generateMatriceFXManuel(f2, tab)
    #print("f1_val=",f1_val)
    #print("f2_val=", f2_val)
    #Courbe représentative de 𝑓1: épaisseur du trait égale à 3, couleur bleuer
    plt.plot(tab, f1_val,'b', 1)
    #Courbe représentative de 𝑓2: points non reliés représentés par *, couleur rouge
    plt.plot(tab, f2_val, 'r*')
    #Afﬁcher le titre : « Tracé de fonctions »
    plt.title('Tracé de fonctions')
    # Afﬁcher « x » pour l’axe des abscisses et « y » pour l’axe des ordonnéesr
    plt.xlabel('x')
    plt.ylabel('y')
    #Axe des x compris entre 0 et 2, axe des y compris entre 0 et 1,5
    plt.xlim(0, 2)
    plt.ylim(0, 1.5)
    plt.grid()
    plt.show()


draw_f1_and_f2()
print("END")

# Version prof
#def corrige():


