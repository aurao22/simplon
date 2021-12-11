"""
1. Définir la classe Point contenant les attributs x et y (coordonnées)
   Implémenter une méthode permettant un affichage sous forme de string du type "(x, y)"
   Implémenter la  méthode __del__ qui affichera "Point détruit"

2. Définir la classe Rectangle
   Pour définir un rectangle, nous spécifions sa largeur, sa hauteur et précisons la position du coin supérieur gauche.
   Une position est définie par un point (coordonnées x et y).
   On s'assurera dans le constructeur que largeur et hauteur sont supposés être des réels positifs.
   Dans le cas contraire ils prendront la valeur 0 par défaut.
   Ils doivent être supérieurs ou égaux à zéro, sinon on mets la valeur 0 par défaut.
   Les modifications sont possibles, on s'assurera simplement que le réel passé est positif,
   dans le cas contraire on ne fera pas de modification.
   Implémenter une méthode permettant un affichage sous forme de string du type "L: 50, H: 36, Coin: (12, 25)"
   Implémenter la  méthode __del__ qui affichera "Rectangle détruit"

3. Instancier un objet mon_rectangle de largeur 4, de hauteur 6, et dont le coin supérieur
   gauche se situe au point de coordonnées (2, 10)
   Tester les différentes méthodes codées

4. Nous avons vu plus haut que les fonctions peuvent utiliser des objets comme paramètres.
   Elles peuvent également transmettre une instance comme valeur de retour.
   Définir la fonction trouveCentre() qui est appelée avec un argument de type Rectangle
   et qui renvoie un objet Point, lequel contiendra les coordonnées du centre du rectangle.

   Tester cette fonction en utilisant l’objet mon_rectangle défini
"""
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


mon_rectangle = Rectangle(4, 6, Point(2, 10))
mon_rectangle.affiche()
print(mon_rectangle.calcul_aire(), "m²")
print("Le centre du rectangle est",trouve_centre(mon_rectangle))

mon_rectangle.hauteur = -5
mon_rectangle.affiche()
mon_rectangle.largeur = -2
mon_rectangle.affiche()











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

def aire_totale(liste_formes):
    aire_tot = 0
    for forme in liste_formes:
        aire_tot += forme.calcul_aire()
    return aire_tot