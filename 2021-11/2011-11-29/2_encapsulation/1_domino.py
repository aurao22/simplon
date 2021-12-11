"""
Domino
Définissez une classe Domino qui permette d’instancier des objets simulant les pièces d’un jeu de dominos.
1. Le constructeur de cette classe initialisera les valeurs des points présents sur les deux faces A et B du domino
   (valeurs par défaut = 0).
2. Deux autres méthodes seront définies :
. une méthode affiche_points() qui affiche les points présents sur les deux faces
. une méthode valeur() qui renvoie la somme des points présents sur les 2 faces.
"""
class Domino:
    """  """

    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b

    def affiche_points(self):
        print(f"face A : {self.a} face B : {self.b}")

    def valeur(self):
        return self.a + self.b



d1 = Domino(2, 6)
d2 = Domino(4, 3)
d1.affiche_points()
d2.affiche_points()

print(f'total des points : {d1.valeur() + d2.valeur()}')
