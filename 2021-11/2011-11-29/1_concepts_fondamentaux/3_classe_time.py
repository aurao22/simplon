"""
Classe Time
1.
. Définir une nouvelle classe Time, qui nous permettra d’effectuer toute une série d’opérations
sur des instants, des durées, etc.
. Ajouter les attributs pays, heure, minute et seconde.
. Créer un objet instant qui contient une heure particulière.
. Ecrivez une fonction affiche_heure() qui permet de visualiser le contenu d’un objet de
  la classe Time sous la forme conventionnelle ”heure : minute : seconde”.
. Appliquer à l’objet instant la fonction affiche_heure()
2.
. Encapsuler la fonction affiche_heure() dans la classe Time(méthode affiche)
. Instancier un objet maintenant
. Tester la méthode
! Une fonction qui est ainsi encapsulée dans une classe s’appelle une méthode.
"""

class Time:
    def __init__(self, pays, heure, minute, seconde):
        self.pays = pays
        self.heure = heure
        self.minute = minute
        self.seconde = seconde

    def affiche(self):
        print(f"{self.heure}:{self.minute}:{self.seconde}")


instant = Time("France", 15,15,15)

def affiche_heure(heure):
    print(f"{heure.heure}:{heure.minute}:{heure.seconde}")

affiche_heure(instant)

maintenant = Time("France", 15,17,15)
maintenant.affiche()

# Correction Pierre
# class Time:
#     def __init__(self, pays, heure, minute, seconde):
#         self.pays = pays
#         self.heure = heure
#         self.minute = minute
#         self.seconde = seconde

#     # def affiche(self):
#     #     print(f'{self.heure:02d}:{self.minute:02d}:{self.seconde:02d}')


# maintenant = Time('France', 9, 45, 3)


# def affiche_heure(instant):
#     print(f'{instant.heure:02d}:{instant.minute:02d}:{instant.seconde:02d}')


# affiche_heure(maintenant)