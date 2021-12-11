"""
https://www.freecodecamp.org/news/python-property-decorator/

On souhaite écrire une classe `Hotel`. Pour cet exercice, un hôtel sera caractérisé par :
– son nom : une chaîne de caractères
– son nombre de chambres : un entier strictement positif

Le nombre de chambres est toujours connu dès la création d’un objet de type Hotel.
Il doit être possible de créer un objet Hotel en précisant son nom ou sans le préciser.
Le nom de l’hôtel doit pourvoir être modifié.
Une tentative de modification du nombre de chambre doit générer un erreur.

Enfin, sur tous les objets Hotel, il faut pouvoir :
– accéder aux valeurs de chacun des attributs ;
– modifier les attributs qui doivent pouvoir être modifiés ;
– récupérer une représentation textuelle de l’état de l’hôtel (méthode __str__())

Ecrire la classe correspondante et un ensemble de test détaillant les différentes configurations possibles
"""


class Hotel:
    def __init__(self, nombre_de_chambres, nom=None):
        self._nombre_de_chambres = nombre_de_chambres
        self.nom = nom

    @property
    def nombre_de_chambres(self):
        return self._nombre_de_chambres

    @nombre_de_chambres.setter
    def nombre_de_chambres(self, nouveau_nombre_de_chambres):
        raise AttributeError('Attibut nombre_de_chambres défini une fois pour toutes dans la méthode init()')

    def __str__(self):
        if self.nom == None:
            texte = 'Hotel'
        else:
            texte = 'Hotel ' + self.nom
        texte = texte + ' , ' + str(self._nombre_de_chambres) + ' chambres'
        return texte


mon_hotel = Hotel(nombre_de_chambres=10)
print('nombre_de_chambres dans mon_hotel: ', mon_hotel.nombre_de_chambres)
# mon_hotel.nombre_de_chambres = 11
print(mon_hotel)

mon_second_hotel = Hotel(nombre_de_chambres=100, nom='Le Majestic')
print(mon_second_hotel)
# # mon_second_hotel.nombre_de_chambres = 200  # lève une exception AttributeError
