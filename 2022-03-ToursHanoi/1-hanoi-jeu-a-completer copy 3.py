from fonction_affiche_hanoi import affiche

def display(state, status="", nb_disques=4):


    affiche(state, n=nb_disques)

    # (tour1, tour2, tour3) = state
    # nb_disques
    # print("--------------------"+status+"-----------------------------")
    # print(f"|         |           |       ")
    # for i in range(nb_disques, -1, -1):
    #     t1 = tour1[i]*"-" if len(tour1) > i else "|"
    #     t2 = tour2[i]*"-" if len(tour2) > i else "|"
    #     t3 = tour3[i]*"-" if len(tour3) > i else "|"
    #     print(f"{t1:8}  {t2:10}  {t3:10}")
    # print(f"______     ______     ______  ")
    # print(f"Tour 1     Tour 2     Tour 3  ")


def init(n):
    """
    Définition de la position initiale
    entrée:
        n :  nombre de disques
    sortie:
        state : position de jeu sous la forme d'un tuple de 3 listes
                tour_1, tour_2, tour_3
    """
    tour1 = [*range(n, 0, -1)]
    tour2 = []
    tour3 = []

    # for i in range(n-1, -1, -1):
    #     tour1.append((i+1))
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
    if state[start] is not None and len(state[start])>0:
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

def get_combinaisons(excluded_moves=[]):

    possibilities = set()

    for i in range(0, 3):
        for j in range(0,3):
            if i != j:
                move = (i,j)
                if move not in excluded_moves:
                    possibilities.add((i, j))

    return possibilities


from random import randint

def search_next_move_random(state, excluded_moves=[]):
    start = -1
    # valeur par défaut puisque c'est la cible
    end = 3

    possibilities = list(get_combinaisons(excluded_moves=excluded_moves))
    finish = False
    found = False
    while not finish:
        i = randint(0, (len(possibilities)-1))
        start, end = possibilities[i]
        del possibilities[i]
        move = (start, end)
        found = is_valid(state, move)
        finish = found or len(possibilities) == 0
    
    if not found:
        return None
    
    return (start, end)


def get_tours():
    return {0,1,2}


def hanoi(state, nb=0, success_states_moves=[], fail_staites_moves=[]):
    if (is_end(state)):
        return state, all_states, nb

    found = False

    moves = []
    while found:
        move = search_next_move_random(state=state, excluded_moves=moves)
        if move is not None:
            nb += nb
            state = successor(state, move)
            all_states.append(state)
            return hanoi(state=state, nb=nb, all_states=all_states)
    


def search_next_move_ia_old(state, nb_disques, last_move):
    start = 0
    target = 2

    state = hanoi(state, nb_disques)

    state = hanoi(state, nb_disques, a=0, b=1, c=2)
    display(state=state, nb_disques=nb_disques)
    state = hanoi(state, nb_disques, a=0, b=1, c=2)

    # valeur par défaut puisque c'est la cible
    end = target
    maxi = target
    mini = 0
    midlle = 1

    # On définit les tours qui ont le mini et le maxi
    for i in range(0, 3):
        if len(state[i])>0:
            if len(state[mini]) > 0 and state[mini][-1]>state[i][-1]:
                mini = i
            elif len(state[maxi])>0 and state[maxi][-1] < state[i][-1]:
                maxi = i
    
    tours = get_tours()
    tours.remove(mini)
    tours.remove(maxi)
    midlle = list(tours)[0]

    # Optimiser les mouvements    
    # Est-ce que le maxi peut aller à la tour cible
    move = False
    finish = False

    if maxi != target:
        i = target
        while move or finish:
            if i != maxi:
                if is_valid(state, (maxi, i)):
                    start = maxi
                    end = i
                    move = True
            i -= 1
            finish = i < 0
        if not move:
            pass
    
    elif maxi != target and is_valid(state, (maxi, target-1)):
        start = maxi
        end = target
    else:
        # est-ce que le mini est sur la cible ?
        if last_move[1] == mini:
            tours = get_tours()
        if mini == target:
            # est-ce qu'on peut bouger le mini sur le middle pour bouger le maxi après ?
            last_move
            pass
    dest = {maxi, midlle}
    started = {mini, midlle, maxi}

    # on les mouvements possibles
    if len(state[mini]) > 1:
        start = mini
        end = state[midlle]
        dest.remove(midlle)
    else:
        start = midlle
        end = state[maxi]
        dest.remove(maxi)
    
    # génération des coups possible
    if not is_valid(state, (start, end)):
        pass

    return (start, end)

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
    mode = input("Mode : manuel, random, auto :")
    modes = {'manuel', 'auto', 'random'}
    state = init(nb_disques)
    i = 0
    exit = False
    if mode in modes:
        while not is_end(state) and not exit:
            display(state, status="", nb_disques=nb_disques)
            start = 0
            end = 0
            if "manuel" in mode:
                try:
                    start = int(input("Tour de départ   :"))
                    end =   int(input("Tour destination:"))
                    start -= 1
                    end -= 1
                except:
                    if "exit" in saisie:
                        exit = True
                    else:
                        print("Erreur de saisie")
            elif "auto" in mode or "random" in mode:
                if "random" in mode:
                    move = search_next_move_random(state=state)
                else:
                    state = hanoi(state, nb_disques, a=0, b=1, c=2)
                    display(state=state, nb_disques=nb_disques)

                if move is None:
                    print("Aucun mouvement possible")
                    break
                else:
                    (start, end) = move
                saisie = input("Next move (enter) or quit (exit)")
                if "exit" in saisie:
                    exit = True
            else:
                break

            state = successor(state, (start, end))
            i += 1
        display(state, status="")
        print("Vous avez gagné en", i, "coups")
    else:
        print("Interruption du jeu")

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


if __name__ == ('__main__'):
    play()