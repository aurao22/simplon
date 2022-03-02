
def display(state, status="", nb_disques=4):
    (tour1, tour2, tour3) = state
    nb_disques
    print("--------------------"+status+"-----------------------------")
    print(f"|         |           |       ")
    for i in range(nb_disques, -1, -1):
        t1 = tour1[i]*"-" if len(tour1) > i else "|"
        t2 = tour2[i]*"-" if len(tour2) > i else "|"
        t3 = tour3[i]*"-" if len(tour3) > i else "|"
        print(f"{t1:8}  {t2:10}  {t3:10}")
    print(f"______     ______     ______  ")
    print(f"Tour 1     Tour 2     Tour 3  ")


def init(n):
    """
    Définition de la position initiale
    entrée:
        n :  nombre de disques
    sortie:
        state : position de jeu sous la forme d'un tuple de 3 listes
                tour_1, tour_2, tour_3
    """
    tour1 = []
    tour2 = []
    tour3 = []

    for i in range(n-1, -1, -1):
        tour1.append((i+1))
    return (tour1, tour2, tour3)


def is_valid(state, move):
    """
    On s'assure que pour la position `state` donnée le déplacement `move` respecte bien les règles
    entrée:
        state
        move est un tuple de deux entiers:
            start, end
    sortie:
        booléen
    """
    start, end = move
    
    # >>> Début de votre code
    # On vérifie que la tour initiale n'est pas vide
    if state[start] is not None:
        # On vérifie que le disque sur lequel on pose le disque déplacé
        # est bien plus grand
        last_dest_elm = state[end][-1] if len(state[end])>0 else None
        elm_to_move = state[start][-1] if len(state[start])>0 else None
        return last_dest_elm is None or (elm_to_move is not None and last_dest_elm > elm_to_move)
    # Les coordonnées des tours seront validées dans l'interface du jeu
    return False
    #     Fin de votre code <<<


def successor(state, move):
    """
    Détermine la position finale après le déplacement `move`
    entrée:
        state
        move
    sortie:
        state
    """
    # >>> Début de votre code
    # On vérifie que le disque sur lequel on pose le disque déplacé est bien plus grand
    if is_valid(state, move):
        start, end = move

        elm_to_move = state[start][-1]
        # Suppression de l'élément de la tour d'origine
        del state[start][-1]
        # Ajout de l'élément dans la tour destination
        state[end].append(elm_to_move)
    return state    
    #     Fin de votre code <<<

def is_end(state):
    """
    Vérification que la position est gagnante
    entrée:
        state
        n
    sortie:
        booléen
    """
    # >>> Début de votre code
    return len(state[0]) == 0 and len(state[1]) == 0
    #     Fin de votre code <<<

# ----------------------------------------------------------------------------------
#                        PLAY
# ----------------------------------------------------------------------------------
# ###### Le jeu ##### #

# Devra comporter entre autre les éléments suivants
# Interface utilisateur
# Vérification de la cohérence des entrées
# Détermination de la position suivante

# >>> Début de votre code
def play(nb_disques = 3):
    nb_disques = int(input("Nombre de disques   :"))
    state = init(nb_disques)
    i = 0
    while not is_end(state):
        try:
            display(state, status="", nb_disques=nb_disques)
            start = int(input("Tour de départ   :"))
            start -= 1
            end =   int(input("Tour destination:"))
            end -= 1
            state = successor(state, (start, end))
            i += 1
        except:
            print("Erreur de saisie")
    display(state, status="")
    print("Vous avez gagné en", i, "coups")

#     Fin de votre code <<<


# ----------------------------------------------------------------------------------
#                        TESTS
# ----------------------------------------------------------------------------------
def test_init(nb_disques = 3):
    # ###### Test des différentes fonctions ##### #
    state = init(nb_disques)
    print("EXPECTED : ([3, 2, 1], [], [])")
    print("OBTAIN   :", state)
    display(state=state, status="INIT", nb_disques=nb_disques)
    return state

def test_move1(state, nb_disques = 3):
    move = (0, 2)
    state = successor(state, move)
    print("EXPECTED : ([3, 2], [], [1])")
    print("OBTAIN   :", state)
    display(state=state, status="MOVE", nb_disques=nb_disques)
    return state

def test():
    # ###### Test des différentes fonctions ##### #
    nb_disques = 3
    state = test_init(nb_disques)
    test_move1(state, nb_disques)
    # print("------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------")
    state = test_init(nb_disques)
    test_move1(state, nb_disques)

    move = (0, 2)
    state = successor(state, move)
    print("EXPECTED : ([3, 2], [], [1])")
    print("OBTAIN   :", state)
    display(state=state, status="MOVE 1 non effectué")

    # # "Déplacement non valide"

    state = [], [], [3, 2, 1]
    assert is_end(state, 3)
    # # True


play()