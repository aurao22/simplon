from click import Parameter
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

def get_combinaisons(state=None,excluded_moves=[]):

    possibilities = set()
    if (state is not None and not is_end(state)) or state is None:
        for i in range(0, 3):
            for j in range(0,3):
                if i != j:
                    move = (i,j)
                    if move not in excluded_moves:
                        if state is not None :
                            # On vérifie si le coup est valide avant de l'ajouter
                            if is_valid(state, move):
                                possibilities.add((i, j))
                        else:
                            possibilities.add((i, j))

    return possibilities


from random import randint

def search_next_move_random(state, excluded_moves=[]):
    start = -1
    # valeur par défaut puisque c'est la cible
    end = 3

    possibilities = list(get_combinaisons(state=state, excluded_moves=excluded_moves))
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

from copy import deepcopy

def hanoi(state, nb_step=0, success_states_moves=None, previous_move=None):
    if state is None:
        raise AttributeError("state missing")

    if (is_end(state)):
        return (True, nb_step, success_states_moves)

    excluded_moves=[]
    if previous_move is not None:
        excluded_moves = [get_reverse_move(previous_move)]

    possibilities = list(get_combinaisons(state=state, excluded_moves=excluded_moves))

    if success_states_moves is None:
        success_states_moves = []
    
    state_to_test = {}
    if possibilities is not None:
        for move in possibilities:
            if is_valid(state, move):
                current_state = deepcopy(state)
                current_state = successor(current_state, move)
                state_to_test[move] = current_state
                # test des noeuds du même niveau uniquement
                if (is_end(current_state)):
                    success_states_moves.append(move)
                    return (True, nb_step+1, success_states_moves)

        # Aucun noeud du niveau ne termine le jeu    
        # les possibilités non valide ont déjà été retirées de la liste des possibilités
        for move,current_state in state_to_test.items():
            temp = deepcopy(success_states_moves)
            temp.append(move)
            child_end = False
            try:
                (child_end, nb_step_child, success_child_child_moves) = hanoi(current_state, nb_step+1, success_states_moves=temp, previous_move=move)
            except Exception as error:
                print(error)
            if child_end:
                return (True, nb_step_child, success_child_child_moves)

    
    # Cas blocage
    raise Exception("Pas de solution")
       

def get_reverse_move(move):
    return (move[1], move[0])

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
                    if move is None:
                        print("Aucun mouvement possible")
                        break
                    else:
                        (start, end) = move
                else:
                    try:
                        (res, nb_step_child, success_child_child_moves) = hanoi(state, nb_step=0, success_states_moves=None)
                        if res:
                            if success_child_child_moves is not None:
                                for mv in success_child_child_moves:
                                    state = successor(state, mv)
                                    display(state=state, nb_disques=nb_disques)
                                i = (len(success_child_child_moves)-1)
                    except Exception as error:
                        print(error)
                if not is_end(state):
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


def test_combinaisons():
    state = [], [], [3, 2, 1]
    assert len(get_combinaisons(state=state)) == 0

    nb_disques = 3
    state = test_init(nb_disques)
    assert get_combinaisons(state=state) == {(0, 1), (0, 2)}

    state = test_move1(state, nb_disques)

    excluded_moves=[get_reverse_move((0, 2))]
    assert get_combinaisons(state=state, excluded_moves=excluded_moves) == {(0, 1), (2, 1)}
    

if __name__ == ('__main__'):
    play()
    # test_combinaisons()