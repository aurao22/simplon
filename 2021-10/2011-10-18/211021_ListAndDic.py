import numpy as np
from random import randint

#𝑓(𝑥)=𝑠𝑖𝑛(𝑥)−𝑥
def f(x):
    res = np.sin(x) - x
    return res
# Fonction courte
#f=lambda x:np.sin(x) - x

#----------------------------------------
# X1 contenant 20 valeurs régulièrement espacées entre 10 et 30.
# Déﬁnir une liste 𝑌1 contenant 20 valeurs déﬁnies par 𝑌1[𝑖]=𝑓(𝑋1[𝑖]).On utilisera la syntaxe range
X1 = []
Y1 = []
for i in range (10, 30, 1):
    X1.append(i)
    # 𝑌1[𝑖]=𝑓(𝑋1[𝑖])
    Y1.append(f(i))
print("X1 :", X1)
print("----------------------------------------")
print("Y1 :", Y1)
print("---------------------------------------------------------------------------------------------------------")
# 𝑋2 contenant 20 valeurs régulièrement espacésentre 10 et 30.
# Déﬁnir un tableau 𝑌2 contenant 20 valeurs déﬁnies par 𝑌[𝑖]=𝑓(𝑋[𝑖]). On utilisera la syntaxe range
X2= np.arange (10, 30, 1)
Y2= []
for i in X2:
    # 𝑌[𝑖]=𝑓(𝑋[𝑖])
    Y2.append(f(i))
print("X2 :", X2)
print("----------------------------------------")
print("Y2 :", Y2)
print("---------------------------------------------------------------------------------------------------------")
# X3 contenant 20 valeurs régulièrement espacésentre 10 et 30 en utilisant la syntaxe np.linspace.
# Déﬁnir un tableau 𝑌3 contenant 11 valeurs déﬁnies par 𝑌[𝑖]=𝑓(𝑋[𝑖]) sans utiliser la syntaxe range
X3 = np.linspace(10, 30, 20)
Y3 = f(X3)
print("X3 :", X3)
print("----------------------------------------")
print("Y3 :", Y3)