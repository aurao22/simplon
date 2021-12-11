"""
Polymorphisme
Ecrivez une classe Forme qui contiendra une méthode calcule_aire.
Faites-en hériter la classe Carre contenant un attribut `cote`
idem pour la classe Cercle contenant un attribut `rayon`
Rédéfinir la méthode calcul_aire pour les classes Carre et Cercle
Définir une fonction calcule_aire_totale qui à partir d’un tableau de `Forme`s calcule l’aire totale
"""
# utilisation de math.pi pour le nombre Pi
import math


class Forme():
   def __init__(self):
      pass

   def affiche(self):
      print(self)
   
   def calcul_aire(self):
      pass


class Point(Forme):
    
   def __init__(self, x=0, y=0):
      self.x = x
      self.y = y

   def __del__(self):
      print("Point",self,"détruit")
   
   def __str__(self):
      return f"({self.x},{self.y})"


class Rectangle(Forme):

   def __init__(self, largeur=0, hauteur=0, point=None):
      # Appel du setter pour contrôle de la valeur
      if hauteur < 0:
         hauteur = 0
      if largeur < 0:
         largeur = 0
      self.largeur = largeur
      # Appel du setter pour contrôle de la valeur
      self.hauteur = hauteur
      self.point = point
      if self.point is None:
         self.point = Point(0,0)

   @property
   def hauteur(self):
      return self._hauteur

   @hauteur.setter
   def hauteur(self, hauteur):
      if hauteur > 0:
         self._hauteur = hauteur

   @property
   def largeur(self):
      return self._largeur

   @largeur.setter
   def largeur(self, largeur):
      if largeur > 0:
         self._largeur = largeur

   def calcul_aire(self):
      return self.largeur*self.hauteur

   def __str__(self):
      return f"L: {self.largeur}, H: {self.hauteur}, Coin: {self.point}"

   def __del__(self):
      print("Rectangle",self,"détruit")


def trouve_centre(mon_rectangle):
   centre = Point()
   if mon_rectangle is not None:
      centre.x = (mon_rectangle.point.x + (mon_rectangle.largeur /2))
      centre.y = (mon_rectangle.point.y + (mon_rectangle.hauteur /2))
   return centre


# mon_rectangle = Rectangle(4, 6, Point(2, 10))
# mon_rectangle.affiche()
# print(mon_rectangle.calcul_aire(), "m²")
# print("Le centre du rectangle est",trouve_centre(mon_rectangle))

# mon_rectangle.hauteur = -5
# mon_rectangle.affiche()
# mon_rectangle.largeur = -2
# mon_rectangle.affiche()


class Carre(Forme):
    def __init__(self, cote):
        Forme.__init__(self)
        self.cote = cote

    def calcul_aire(self):
        return self.cote**2

class Cercle(Forme):
    def __init__(self, rayon):
        Forme.__init__(self)
        self.rayon = rayon

    def calcul_aire(self):
        return self.rayon**2 * math.pi

def calcule_aire_totale(liste_formes):
    aire_tot = 0
    for forme in liste_formes:
        aire_tot += forme.calcul_aire()
    return aire_tot


cercle_1 = Cercle(1)
print('cercle_1 : ', cercle_1.calcule_aire_totale())

carre_2 = Carre(2)
print('carre_2 : ', carre_2.calcule_aire_totale())


# Polymorphisme
formes = [cercle_1, carre_2]
aire_totale = calcule_aire_totale(formes)
assert calcule_aire_totale(formes) == 7.141592653589793
print('Aire totale pour l\'ensemble des formes', aire_totale)
