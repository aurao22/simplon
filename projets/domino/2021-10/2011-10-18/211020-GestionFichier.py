# A vous de jouer !
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as lina

# Fichier de Données
fName = "C:/Users/User/Documents/SIMPLON/workspace/jour1/2110210_data.txt"
f = open(fName, 'w')
f.write('25 ;10.9\n')
f.write('20 ;9.3\n')
f.write('15 ;8.2\n')
f.write('12 ;7.5\n')
f.write('9 ;6.2\n')
f.write('6 ;5.8\n')
f.write('3 ;4.2\n')
f.write('0 ;3.9\n')
f.write('-3 ;2.8\n')
f.write('-6 ;2\n')
f.close()

# Lire le fichier
f = open(fName, 'r')

lines = f.readlines()
# fermez le fichier après avoir lu les lignes
f.close()

# Initialisation des listes
abscisses = np.array([])
ordonnees = np.array([])

S1 = 0
S2 = 0
n = len(lines)

# Itérer sur les lignes
for line in lines:
    coord = line.strip().split(sep=' ;')
    # print(coord)
    abscisses = np.append(abscisses, float(coord[0]))
    ordonnees = np.append(ordonnees, float(coord[1]))
    S1 += float(coord[0]) * float(coord[1])
    S2 += float(coord[0]) ** 2
# -----------
# Calcul de la version prédictive
S1 = S1 * n
S2 = S2 * n
SX = np.sum(abscisses)
SY = np.sum(ordonnees)
print("-----------")
print(abscisses)
print(ordonnees)
a = (S1 - (SX * SY)) / (S2 - SX ** 2)
# print("a :", a)
# 𝑏=𝑦⎯⎯⎯−𝑎𝑥⎯⎯⎯ en notant 𝑥⎯⎯⎯ la moyenne des 𝑥𝑖 et 𝑦⎯⎯⎯ la moyenne des 𝑦𝑖
b = np.mean(ordonnees) - a * np.mean(abscisses)
# print("b :", b)
# print("-----------")
# y modélisé = ax + b
ordonnees2 = np.array([])

for x in abscisses:
    ordonnees2 = np.append(ordonnees2, a * x + b)

plt.plot(abscisses, ordonnees2)
plt.show()
plt.plot(ordonnees, ordonnees2)
# plt.plot(abscisses, ordonnees2, 'ro')
plt.show()

print("----------- CONTROL ---------------")
control = 0
i = 0
for i in range(0, n):
    control += (ordonnees[i] - ordonnees2[i]) ** 2
print("control:", control)
control = math.sqrt(control)
print("control:", control)

# ----

dist = lina.norm(ordonnees - ordonnees2, 2)
print(dist)