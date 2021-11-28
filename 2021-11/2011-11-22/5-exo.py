# Exercice 1.1. Classe et instances
# Ecrire la classe ´ Voiture en python.
# Ecrire son son contructeur avec la possibilit´e´ de pr´eciser les valeurs de certains attributs `a l’instanciation.
# Instancier 3 objets issus de la classe Voiture

class Voiture:
    
    def __init__(self, marque, modele, couleur):
        self.marque = marque
        self.modele = modele
        self.couleur = couleur


ma_voiture = Voiture("Dacia", "Duster", "Gris comète")

print(f"ma_voiture est une {ma_voiture.marque}, {ma_voiture.modele} {ma_voiture.couleur}")


# Exercice 1.2. Manipulation d’instances 
# Cr´eer un tableau contenant les instances de la classe Voiture. 
# A!cher la valeur de l’attribut couleur de chaque instance en parcourant ce tableau
ta_voiture = Voiture("Renault", "Scénic", "Blanc")
sa_voiture = Voiture("Vélo", "de ville", "bleu")
leur_voiture = Voiture("DAF", "Car", "Rouge")

liste_voiture = [ma_voiture, ta_voiture, sa_voiture, leur_voiture]

for vehicules in liste_voiture:
    print(vehicules.couleur)


# Exercice 2.1. Impl´ementez la Classe Fichier. 
# On peut parler des fichiers (en g´en´eral) sans faire r´ef´erence `a un fichier particulier, 
# il s’agit donc d’une classe. 
# Un fichier est caract´eris´e par son nom, sa taille, sa date de cr´eation et sa date de modification. 
# Un fichier peut s’ouvrir et se fermer.

class Fichier:

    def __init__(self, nom, taille, creation, modification):
        self.nom = nom
        self.taille = taille
        self.creation = creation
        self.modification = modification

    def ouvrir(self):
        pass

    def fermer(self):
        pass


from abc import ABC, abstractmethod
# Exercice 2.2. Gestion du stock
# Un Article du stock est d´efini par 4 champs :
# . sa r´ef´erence (num´ero)
# . sa d´esignation (texte)
# . son prixHT
# . sa quantit´e (nombre d’articles disponibles)
# Pour manipuler ces champs, les services suivants sont fournis :
# . prixTTC
# . prixTransport (taxe 5% prixHT)
# . retirer
# . ajouter

class Article:

    def __init__(self, reference, designation, prix_ht=0, quantite=0) -> None:
        self.reference = reference
        self.designation = designation
        self.prix_ht = prix_ht
        self.quantite = quantite

    def prix_ttc(self):
        return self.prix_ht*1.20

    def prix_transport(self):
        return self.prix_ht*0.05

    def retirer(self, quantite):
        if self.quantite >= quantite:
            self.quantite -= quantite
        else:
            return None
        return self.quantite

    def ajouter(self, quantite):
        self.quantite += quantite
        return self.quantite


chaussure = Article(45, "Ma chaussure", 55, 10)
print(f"{chaussure.designation} reste en stock {chaussure.retirer(2)}")
print(f"prix {chaussure.prix_ht} HT")
print(f"transport {chaussure.prix_transport()}")
print(f"prix {chaussure.prix_ttc()} TTC")
print(f"{chaussure.designation} en stock {chaussure.ajouter(2)}")


# Exercice 4.1. H´eritage
# Ecrire une classe Animal avec 2 attributs (couleur, poids) et 2 m´ethodes (afficheDureeDeVie et crier)
# Ecrire une classe Chien (classe enfant de Animal)
# . Instancier m´edor instance de Chien
# . Faite le crier
# . Modifier la classe Chien pour pr´eciser
# ⇧ que sa dur´ee de vie est de 15ans
# ⇧ et son poids de 30kg par d´efaut

# hériter de ABC(Abstract base class)
# class Animal(ABC):
#     # Constructeur
#     def __init__(self, couleur, poids):
#         self.couleur = couleur
#         self.poids = poids
    
#     @abstractmethod # un décorateur pour définir une méthode abstraite
#     def afficheDureeDeVie(self):
#         pass

#     @abstractmethod
#     def crier(self):
#         pass


class Animal:
    # Constructeur
    def __init__(self, couleur, poids):
        self.couleur = couleur
        self.poids = poids
        self.nouriture = "ours"
    
    # un décorateur pour définir une méthode abstraite
    def afficheDureeDeVie(self):
        pass

    def crier(self):
        pass

    def manger(self):
        print("manger un ", self.nouriture)


    
class Chien(Animal):
    # Constructeur
    def __init__(self, couleur, poids=30):
        super().__init__(couleur, poids)
    
    def afficheDureeDeVie(self):
        print("Un chien a une espérance de vie 10 à 13 ans en moyenne, cet âge varie énormément en fonction de la taille de l'animal")

    def crier(self):
        return "Aboiement"

    def __del__(self):
        print("del")


class Lion(Animal):
    # Constructeur
    def __init__(self, couleur, poids):
        super().__init__(couleur, poids)

    def afficheDureeDeVie(self):
        print("Lion a une espérance de vie 10 à 15 ans à l'état sauvage")

    def crier(self):
        return "Rugissement"
    
    def __del__(self):
        print("del")


class Panda(Animal):
    # Constructeur
    def __init__(self, couleur, poids):
        super().__init__(couleur, poids)

    def afficheDureeDeVie(self):
        print("Panda géant a une espérance de vie 20 ans à l'état sauvage")

    def crier(self):
        return "Bêlement"

    def __del__(self):
        print("del")


animaux = [Chien("Noir et feu", 25), Lion("Jaune sable", 300), Panda("Noir et Blanc", 70)]

for animal in animaux:
    print(f"{type(animal).__name__} : {animal.crier()}")
    animal.manger()


# Exercice 4.2. Gestion du stock
# La classe Article contient un prixHT. Ajouter une classe Vetement contenant les mˆemes informations et services que la classe Article, 4
# avec en plus les attributs taille et coloris. Ajouter une classe ArticleDeLuxe contenant les mˆemes
# informations et services que la classe Article, # si ce n’est une nouvelle d´efinition de la m´ethode
# prixTTC (taxe di↵´erente).

class Vetement(Article):

    def __init__(self, reference, designation, taille, coloris, prix_ht=0, quantite=0) -> None:
        super().__init__(reference, designation, prix_ht, quantite)
        self.taille = taille
        self.coloris = coloris


class ArticleDeLuxe(Vetement):

    def __init__(self, reference, designation, taille, coloris, prix_ht=0, quantite=0) -> None:
        super().__init__(reference, designation, taille, coloris, prix_ht, quantite)

    def prix_ttc(self):
        return self.prix_ht*2.20


# Exercice 5.1. classe Autonombile
# Impl´ementer la classe Autonombile `a partir des classes Carrosserie, Siege, Roue et Moteur.
