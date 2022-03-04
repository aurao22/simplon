# BFS example
# Breadth First Search
# Parcours en largeur

# On fournit un sommet de départ s
# et le dictionnaire de connexions: Adjency lists (adj)


def bfs(s, adj):
    """
    # s: sommet de départ
    # adj: dictionnaire des listes d'adjacence
    #   adj[sommet] = [liste de sommets connexes]
    """
    level = {s: 0}  # Niveau de profondeur de la recherche
    parent = {s: None}  # parent du sommet
    i = 1
    frontier = [s]  # Liste représentant la frontière en cours d'exploration
    while frontier:
        next_frontier = []  # Prochaine frontière visible
        for u in frontier:
            for v in adj[u]:
                if v not in level:
                    level[v] = i
                    parent[v] = u
                    next_frontier.append(v)
        frontier = next_frontier
        i += 1


# Graphe étudié
#  a - s   d - f
#  |   | / | / |
#  z   x - c - t

# Définir le dictionnaire des adjacences
# >>> Début de votre code
pass
#     Fin de votre code <<<

# Afficher les frontières de différents niveaux
# en appelant la fontion bfs
# "Frontière niveau  1  :  ..."

# >>> Début de votre code
pass
#     Fin de votre code <<<

# intervertir t et d
# Modifier la fonction pour qu'elle s'arrêtre dès qu'on atteind t
# >>> Début de votre code
pass
#     Fin de votre code <<<

"""
Avant de généraliser faire le cas deux disques:
1- Comment représenter les sommets
2- Faire le graphe du jeu avec deux disques et écrire le dictionnaire des adjacences
3- Faire tourner l’algorithme BFS avec votre modélisation pour explorer l’ensemble du graphe
4- Sortir le chemin qui va de la solution à la position de départ
5- Faire jouer l’ordinateur tout seul dans ce cas de figure
"""
