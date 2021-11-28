with open("211108_dico.txt", "r") as fichier:
    dico = fichier.readlines()
    dico = [mot.rstrip() for mot in dico]

voyelles = "aàâiïîoôuùûy"
mot_avec_e = 0
for mot in dico:
    lettres = set(mot)
    not set(lettres).issubset(set(voyelles))
    if 'e' in lettres and voyelles not in lettres:
        mot_avec_e += 1

pourcentage = mot_avec_e / len(dico) * 100
print(pourcentage)