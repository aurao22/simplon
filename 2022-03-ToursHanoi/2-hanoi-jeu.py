import os


def init(n):
    """
    Définition de la position initiale
    entrée:
        n :  nombre de disques
    sortie:
        state : position de jeu sous la forme d'un tuple de 3 listes
                tour_1, tour_2, tour_3
    """
    state = [i for i in range(n, 0, -1)], [], []
    return state


def is_valid(state, move):
    """
    On s'assure que pour la position `state` donnée
    le déplacement `move` respecte bien les règles
    entrée:
        state
        move est un tuple de deux entiers:
            start, end
    sortie:
        booléen
    """
    start, end = move
    start_tower = state[start]
    end_tower = state[end]
    valid = True
    if len(start_tower) == 0:
        # On vérifie que la tour initiale n'est pas vide
        valid = False
    else:
        disk = start_tower[-1]

        if len(end_tower) != 0:
            # On vérifie que le disque sur lequel on pose le disque déplacé
            # est bien plus grand
            if disk > end_tower[-1]:
                valid = False
    if not valid:
        print("Déplacement non valide")
    return valid


def successor(state, move):
    """
    Détermine la position finale après le déplacement `move`
    entrée:
        state
        move
    sortie:
        state
    """
    if is_valid(state, move):
        start, end = move
        disk = state[start][-1]
        del state[start][-1]
        state[end].append(disk)
    return state


def is_end(state, n):
    """
    Vérification que la position est gagnante
    entrée:
        state
        n
    sortie:
        booléen
    """
    if len(state[2]) == n:
        return True
    else:
        return False


def affiche(state, n):
    os.system('clear')
    for niveau in range(n, 0, -1):
        elt = ''
        for tower in state:
            if niveau <= len(tower):
                i = tower[niveau - 1]
                elt_temp = ' ' * (n - i) + '*' * \
                    (2 * (i - 1) + 1) + ' ' * (n - i) + ' '
                elt += elt_temp
            else:
                elt += ' ' * (2 * (n - 1) + 1) + ' '
        print(elt)
    legende = ' ' * (n - 1) + '1' + ' ' * (n - 1) + ' ' \
        + ' ' * (n - 1) + '2' + ' ' * (n - 1) + ' ' \
        + ' ' * (n - 1) + '3' + ' ' * (n - 1) + ' '
    print(legende)


# ### Le jeu ####


n = 4
state = init(n)
affiche(state, n)


while not is_end(state, n):

    # Interface utilisateur
    wrong_input = True
    while wrong_input:
        entry = input("Tours de départ et d'arrivée : ")
        # Passage aux indices des tuples
        start = int(entry[0]) - 1
        end = int(entry[1]) - 1
        # Vérification de la cohérence des entrées
        if (0 <= start <= 2) and (0 <= end <= 2):
            wrong_input = False
            move = (start, end)
        else:
            print("Entrée non valide")
            wrong_input = True
    # On détermine la position suivante
    state = successor(state, move)
    # print(state)
    affiche(state, n)

print("Gagné")
