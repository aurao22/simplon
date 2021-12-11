"""
Classe Livre
1. Ecrire une classe Livre avec les attributs suivants :
. titre : Le titre du livre,
. auteur : L’auteur du livre,
. prix : Le prix du livre,
. annee : L’année du livre.
2. La classe Livre doit pouvoir être construites avec les possibilités suivantes :
. Livre(),
. Livre(titre),
. Livre(titre, auteur),
. Livre(titre, auteur, prix),
. Livre(titre, auteur, prix, annee)
3. Instancier deux livres de votre choix
4. Créer une fonction qui renvoie vrai si le titre de l’instance passée en paramètre correspond au titre demandé.
"""

class Livre:
    def __init__(self, titre=None, auteur=None, prix=None, annee=None):
        self.titre = titre
        self.auteur = auteur
        self.prix = prix
        self.annee = annee


mon_livre1 = Livre("Mon livre1")
mon_livre2 = Livre("Mon livre2")


# correction Pierre

# class Livre:
#     def __init__(self, titre=None, auteur=None, prix=None, annee=None):
#         self.titre = titre
#         self.auteur = auteur
#         self.prix = prix
#         self.annee = annee


# mon_premier_livre = Livre("Candide", "Voltaire")
# mon_second_livre = Livre("Les neiges bleues", "Piotr Bednarski")


# def est_titre(livre, titre_demande):
#     return livre.titre == titre_demande


# print("Les neiges bleues ", est_titre(mon_second_livre, "Les neiges bleues"))
# print("Les neiges éblouies ", est_titre(mon_second_livre, "Les neiges éblouies"))