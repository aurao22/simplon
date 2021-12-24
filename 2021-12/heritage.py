# Héritage en POO (inheritance)
import math

# ##############################################


class Forme:
    def __init__(self):
        print("Je suis une forme quelconque")


class Rectangle(Forme):
    def __init__(self):
        print("Je suis un rectangle")


class Cercle(Forme):
    def __init__(self):
        print("Je suis un cercle")


# forme = Forme()
rectangle = Rectangle()
cercle = Cercle()

# ##############################################

# class Forme:
#     def __init__(self):
#         print("Je suis une forme quelconque")
#         self.perimetre = None

#     def calcule_perimetre(self):
#         print("Je ne connais pas de formule pour le cas général")
#         return None
#         # raise NotImplementedError()


# class Rectangle(Forme):
#     def __init__(self, longueur, largeur):
#         print("Je suis un rectangle")
#         self.longueur = longueur
#         self.largeur = largeur
#         self.perimetre = None

#     def calcule_perimetre(self):
#         self.perimetre = 2 * (self.longueur + self.largeur)
#         print(f"Perimètre = {self.perimetre}")
#         return self.perimetre

# # class Cercle(Forme):
# #     def __init__(self):
# #         print("Je suis un cercle")


# forme = Forme()
# forme.calcule_perimetre()
# rectangle = Rectangle(1, 2)
# rectangle.calcule_perimetre()

# ##############################################

# class Carre(Rectangle):
#     # pass
#     def __init__(self, cote):
#         print("Je suis un carré")
#         super().__init__(cote, cote)


# # carre = Carre(2, 2)
# carre = Carre(2)
# carre.calcule_perimetre()
