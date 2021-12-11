"""
Héritage simple et multiple dans le monde de l'âge de glace

Définir une classe Mammifere, qui contiendra deux caractéristiques nom et âge

A partir de cette classe, nous pourrons alors dériver une classe Primate, une classe Rongeur,
et une classe Carnivore, qui hériteront toutes les caractéristiques de la classe Mammifere.

Spécificités:
les rongeurs (comme scrat) pourront être initiés avec un nombre de glands
les carnivores pourront être initiés avec une caractéristique supplémentaire
('dents de sabre' pour Diego)


Définir une classe Herbivore dérivé de la classe Primate
Définir une classe Omnivore, qui hérite de la classe Carnivore et de la classe Herbivore

Instance de test à créer:
Mammifère: Manfred
Primate: Tarzan
Rongeur: Srat
Carnivore: Diego
Omnivore : Jane
"""

class Mammifere():

    def __init__(self, nom, age, nb_dents):
        self.nom = nom
        self.age = age
        self.nb_dents = nb_dents
    
    def affiche(self):
        print(self.nom, self.age, self.nb_dents)

class Primate(Mammifere):

    def __init__(self, nom, age, nb_dents):
        Mammifere.__init__(self, nom, age, nb_dents)

class Rongeur(Mammifere):

    def __init__(self, nom, age, nb_dents):
        Mammifere.__init__(self, nom, age, nb_dents)

class Carnivore(Mammifere):

    def __init__(self, nom, age, nb_dents):
        Mammifere.__init__(self, nom, age, nb_dents)

class Belette(Carnivore):

    def __init__(self, nom, age, nb_dents):
        Carnivore.__init__(self, nom, age, nb_dents)
    
class Loup(Carnivore):

    def __init__(self, nom, age, nb_dents):
        Carnivore.__init__(self, nom, age, nb_dents)

class Chien(Carnivore):

    def __init__(self, nom, age, nb_dents):
        Carnivore.__init__(self, nom, age, nb_dents)

class Herbivore(Primate):

    def __init__(self, nom, age, nb_dents, hiberne):
        Primate.__init__(self, nom, age, nb_dents)

class Omnivore(Herbivore, Carnivore):

    def __init__(self, nom, age, nb_dents, hiberne):
        Herbivore.__init__(self, nom, age, nb_dents, hiberne)
        Carnivore.__init__(self, nom, age, nb_dents)

mon = Omnivore("Jane", 45, 58, False)
mon.affiche()
