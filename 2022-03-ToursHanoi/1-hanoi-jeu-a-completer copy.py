
def display(state, status=""):
    (tour1, tour2, tour3) = state
    l = len(tour1)-1
    print("--------------------"+status+"-----------------------------")
    print(f"|         |           |       ")
    for i in range(l, -1, -1):
        t1 = tour1[i][1] if len(tour1[i][1]) > 0 else "|"
        t2 = tour2[i][1] if len(tour2[i][1]) > 0 else "|"
        t3 = tour3[i][1] if len(tour3[i][1]) > 0 else "|"
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
        tour1.append(((i+1), "-"*(i+1)))
        tour2.append(get_empty_disk())
        tour3.append(get_empty_disk())
    return (tour1, tour2, tour3)

def get_empty_disk():
    return ("","")


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
    # >>> Début de votre code
    # On vérifie que la tour initiale n'est pas vide
    if state[move[0]] is not None:
        # On vérifie que le disque sur lequel on pose le disque déplacé
        # est bien plus grand
        last_dest_elm = get_last(state[move[1]])
        elm_to_move = get_last(state[move[0]])
        return last_dest_elm is None or last_dest_elm[0] > elm_to_move[0]
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
        elm_to_move = state[move[0]][-1]
        # Suppression de l'élément de la tour d'origine
        del state[move[0]][-1]
        state[move[0]].append(get_empty_disk())
        # Ajout de l'élément dans la tour destination
        del state[move[1]][-1]
        state[move[1]].insert(0, elm_to_move)
    return state    
    #     Fin de votre code <<<

def is_empty(tour):
    for t in tour:
        if not (t[0] == "" and t[1] == ""):
            return False
    return True

def get_last(tour):
    for t in tour:
        if not (t[0] == "" and t[1] == ""):
            return t
    return None

def is_end(state, n):
    """
    Vérification que la position est gagnante
    entrée:
        state
        n
    sortie:
        booléen
    """
    # >>> Début de votre code
    return is_empty(state[0]) and is_empty(state[1])
    #     Fin de votre code <<<

# ###### Test des différentes fonctions ##### #
# state = init(3)
# print(state)
# display(state=state, status="INIT")
# # # ([3, 2, 1], [], [])

# state = init(3)
# move = (0, 2)
# display(state=state, status="INIT")
# state = successor(state, move)
# display(state=state, status="MOVE 1")
# print(state)
# # # ([3, 2], [], [1])
# print("------------------------------------------------------------------------------")
# print("------------------------------------------------------------------------------")
state = init(3)
display(state=state, status="INIT")
print(state)
move = (0, 2)
state = successor(state, move)
display(state=state, status="MOVE 1")
print(state)
move = (0, 2)
state = successor(state, move)
display(state=state, status="MOVE 1 non effectué")
print(state)
# # "Déplacement non valide"

state = [], [], [3, 2, 1]
print(is_end(state, 3))
# # True


# ###### Le jeu ##### #

# Devra comporter entre autre les éléments suivants
# Interface utilisateur
# Vérification de la cohérence des entrées
# Détermination de la position suivante

# >>> Début de votre code
pass
#     Fin de votre code <<<
