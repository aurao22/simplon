from abc import ABC, abstractmethod

"""
https://www.freecodecamp.org/news/python-property-decorator/

On souhaite écrire une classe `Hotel`. Pour cet exercice, un hôtel sera caractérisé par :
– son nom : une chaîne de caractères
– son nombre de chambres : un entier strictement positif

Le nombre de chambres est toujours connu dès la création d’un objet de type Hotel.
Il doit être possible de créer un objet Hotel en précisant son nom ou sans le préciser.
Le nom de l’hôtel doit pourvoir être modifié.
Une tentative de modification du nombre de chambres doit générer un erreur.

Enfin, sur tous les objets Hotel, il faut pouvoir :
– accéder aux valeurs de chacun des attributs ;
– modifier les attributs qui doivent pouvoir être modifiés ;
– récupérer une représentation textuelle de l’état de l’hôtel (méthode __str__())

Ecrire la classe correspondante et un ensemble de test détaillant les différentes configurations possibles
"""

class Hotel:

    def __init__(self, nb_chambres, nom=None):
        self.nom = nom
        if isinstance(nb_chambres, int) and nb_chambres > 0:
            self._nb_chambres = nb_chambres
        else:
            raise Exception("Le nombre de chambres doit être supérieur à 0")

    @property
    def nb_chambres(self):
        return self._nb_chambres

    @nb_chambres.setter
    def nb_chambres(self, nb_chambres):
        raise Exception("Le nombre de chambres ne peut pas être modifié")


hotel0 = Hotel(-1, "hotel1")
hotel0 = Hotel(None, "hotel1")
hotel1 = Hotel(3, "hotel1")
hotel2 = Hotel(6, "hotel2")
hotel3 = Hotel(9, "hotel3")

hotel1.nb_chambres = 5
