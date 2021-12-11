"""
https://towardsdatascience.com/5-ways-to-control-attributes-in-python-an-example-led-guide-2f5c9b8b1fb0
Example 5: Conditional checking in the init Constructor

CompteBancaire
Définissez une classe CompteBancaire, qui permette d’instancier des objets tels que compte1, compte2, etc.
1. Le constructeur de cette classe initialisera deux attributs nom et solde,
avec les valeurs par défaut `None` et 0.
Lors de l'initialisation d'une instance
Le nom devra être une chaîne de caractères et sera stockée en lettre capitales
Le solde devra être un entier strictement positif

2. Trois autres méthodes seront définies :
. depot(somme) permettra d’ajouter une certaine somme supposée positive au solde
. retrait(somme) permettra de retirer une certaine somme du solde,
  renvoie vrai si l'action est possible (solde final >= 0)
. affiche() permettra d’afficher le nom du titulaire et le solde de son compte.

"""

class CompteBancaire:
    def __init__(self, nom=None, solde=0):
        if isinstance(nom, str):
            self._nom = nom.upper()
        else:
            raise Exception('type de nom incorrect, str demandé')

        if isinstance(solde, int):
            if solde > 0:
                self._solde = solde
            else:
                raise Exception('solde doit un entier strictement positif')
        else:
            raise Exception('type de solde incorrect: int demandé')

    def depot(self, somme):
        self._solde = self._solde + somme

    def retrait(self, somme):
        solde_restant = self._solde - somme
        if solde_restant >= 0:
            self._solde = solde_restant
            return True
        else:
            return False

    def affiche(self):
        print(f'{self._nom} : {self._solde} €')

# compte_pb = CompteBancaire(1,1)
# compte_pb = CompteBancaire('moi','un')
# compte_pb = CompteBancaire('moi',0)

mon_compte = CompteBancaire('toto',100)
mon_compte.depot(10)
retrait_possible = mon_compte.retrait(200)
print(retrait_possible)
mon_compte.retrait(20)
mon_compte.affiche()

# Aides:
# Tester un type en python
# print(isinstance('test', str))
# print(isinstance(None, str))
# print(isinstance('test', int))

# Lever une exeption dans le cas d'un attribut mal renseigné
# raise AttributeError('attribut mal renseigné')
