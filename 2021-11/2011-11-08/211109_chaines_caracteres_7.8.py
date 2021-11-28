# Le mot le plus long
from time import time


def dictionnaire_ordonne():
    # on lit le fichier et on range les mots alphabétiquement selon leur longueur
    fichier = open("211108_dico.txt", "r")
    dict_ord = {}
    for longueur in range(26):
        dict_ord[longueur + 1] = []
    mot = fichier.readline()
    while mot != '':
        mot = mot.strip('\n')
        if '-' in mot:
            mot = mot.replace('-', '')
        if "'" in mot:
            mot = mot.replace("'", '')
        dict_ord[len(mot)].append(mot)
        mot = fichier.readline()
    fichier.close()
    return dict_ord


def lettres_multiples_ok(mot, tirage):
    # teste si chaque lettre figure suffisamment de fois dans le tirage
    for lettre in mot:
        if lettre in tirage:
            tirage.remove(lettre)
        else:
            return False
    return True


def trouver_plus_long_mot(dico, tirage):
    longueur_mot = len(tirage)
    solution = []

    while longueur_mot > 0:
        for mot in dico[longueur_mot]:
            # les lettres du mot sont un sous-ensemble du tirage
            tirage_test = list(tirage)
            if lettres_multiples_ok(mot, tirage_test):
                solution.append(mot)
        if solution != [] or longueur_mot == 1:
            return solution, longueur_mot
        longueur_mot -= 1


def trouver_plus_long_mot2(dico, tirage):
    longueur_mot = len(tirage)
    solution = []

    while longueur_mot > 0:
        for mot in dico[longueur_mot]:
            is_include = True
            for lettre in mot:
                if lettre in tirage:
                    continue
                else:
                    is_include = False
                    break

            if is_include:
                # les lettres du mot sont un sous-ensemble du tirage
                tirage_test = list(tirage)
                if lettres_multiples_ok(mot, tirage_test):
                    solution.append(mot)
        if solution != [] or longueur_mot == 1:
            return solution, longueur_mot
        longueur_mot -= 1


dico = dictionnaire_ordonne()
jouer = True
while jouer:
    tirage = input('Entrez votre tirage : ').lower()
    print(len(tirage), 'lettres')
    t0 = time()
    solution, longueur = trouver_plus_long_mot(dico, tirage)
    if solution == []:
        print('Pas de mot trouvé pour', tirage)
    else:
        t1 = time() - t0
        print(tirage, 'Le script a mis', '{0:.3f}'.format(t1),
              's pour trouver des mots de', longueur, 'lettres')
        print(solution)
    rejouer = input('Rejouer ? (o/n) : ')
    jouer = (rejouer == "o")
