import os

def affiche(state, n):
    """Fonction d'affichage des tours de Hanoï en mode console

    Args:
        state : position de jeu sous la forme d'un tuple de 3 listes
                (tour_1, tour_2, tour_3)
        param2 (int): nombre de disques
    """
    os.system('cls')
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

def hanoi(n,a=1,b=2,c=3, nb=0):
    if (n > 0):
        nb = hanoi(n-1,a,c,b, nb+1)
        # print("Déplace ",a,"sur",c)
        nb = hanoi(n-1,b,a,c, nb+1)
    return nb


if __name__ == ('__main__'):
    state = ([4, 3, 2], [], [1])
    affiche(state, 4)

    nb = hanoi(5)
    print(3, "anneaux fin en ",nb, "coups")





