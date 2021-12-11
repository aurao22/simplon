"""
classe Ensemble
1. Créez une classe Ensemble pour représenter des sous-ensembles de l’ensemble des entiers compris entre 0 et N - 1,
   où N est une constante donnée, avec les méthodes suivantes :
. contient(i) : test d’appartenance,
. ajoute(i) : ajout d’un élément à l’ensemble,
. __str__() : transformation de l’ensemble en une chaîne de caractères de la forme {5 , 6, 10, 15, 20, 21, 22, 30}

Les éléments d’un ensemble seront mémorisés dans une variable d’instance de type set().

Pour essayer la classe Ensemble, écrivez un programme déterminant les nombres différents contenus dans une suite
de nombres lue sur l’entrée standard.

2. Modifiez la classe précédente afin de permettre la représentation d’ensembles d’entiers quelconques,
c’est-à-dire non nécessairement compris entre 0 et une constante N connue à l’avance.
Faites en sorte que l’interface de cette classe soit identique à celle de la version précédente
 qu’il n’y ait rien à modifier dans les programmes qui utilisent la première version de cette classe.
"""

class Ensemble:
    def __init__(self, ensemble, borne=None):
        self.borne = borne
        self.ensemble = set()
        if ensemble is not None:
            for elt in ensemble:
                self.ajoute(elt)

    def cardinal(self):
        return len(self.ensemble)

    def contient(self, i):
        return i in self.ensemble
    
    def ajoute(self, i):
        if self.borne is not None:
            if -1 < i < self.borne :
                self.ensemble.add(i)
            else:
                print(i, "not in 0 -", self.borne)
        else:
            self.ensemble.add(i)
    
    def toString(self):
        res = "{"
        for el in sorted(self.ensemble):
            if len(res) > 1:
                res += ","
            res += str(el) 
        res = res+ "}"
        return res


print("Cas d'un ensemble borné")
mon_ensemble = Ensemble({-10, -3, 5, 6, 10, 15, 20, 21, 22, 30}, 20)
print(mon_ensemble.toString())
mon_ensemble.ajoute(100)
mon_ensemble.ajoute(-1)
mon_ensemble.ajoute(1)
print(mon_ensemble.toString())

print("Cas d'un ensemble non borné")
mon_ensemble = Ensemble({-10, -3, 5, 6, 10, 15, 20, 21, 22, 30})
print(mon_ensemble.toString())
mon_ensemble.ajoute(100)
mon_ensemble.ajoute(-1)
mon_ensemble.ajoute(1)
print(mon_ensemble.toString())


# Correction Pierre

class Ensemble2:
    def __init__(self, e, N):
        """
        Ensemble d'entiers compris entre 0 et N - 1
        e : set()
        N : integer
        """
        self.N = N
        self.e = e.intersection(set(range(self.N)))

    def contient(self, i):
        return i in self.e

    def ajoute(self, i):
        if 0 <= i < self.N:
            self.e = self.e.union({i})

    def __str__(self):
        representation = str(sorted(self.e))  # Par ordre croissant mais sous forme de liste avec []
        representation = representation.replace('[', '{')
        representation = representation.replace(']', '}')
        return representation

print("----------------------------------")
mon_ensemble = Ensemble2({-10, -3, 5, 6, 10, 15, 20, 21, 22, 30}, 20)
print(mon_ensemble)
