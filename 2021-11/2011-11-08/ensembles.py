# Ensembles

ens1 = set([1, 2, 3, 4, 5, 6])
ens2 = set([5, 6, 7, 8])
ens3 = set([2, 4, 6])

# union
print(ens1 | ens2)
# intersection
print(ens1 & ens2)
# différence symétrique
print(ens1 ^ ens2)
# différence
print(ens1 - ens2)

# inclusion
print(ens2.issubset(ens1))
print(ens3.issubset(ens1))

# transformation en une liste
liste1 = list(ens1)
print(liste1)
