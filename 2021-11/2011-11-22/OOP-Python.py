# Progammation orientée objet en python

# Objet
# > propriétés
# > comportements

# ordinateur, mac/windows, taille du  disque, fréquence...
# s'allumer, fonctionner, se planter

class Ordinateur:
    pass


mon_ordi = Ordinateur()
mon_portable = Ordinateur()

print(mon_ordi)
print(mon_portable)
print(mon_ordi == mon_portable)


# # Variable de classe
# class Ordinateur:
#     processeur_principal = 'CPU'


# mon_ordi = Ordinateur()
# mon_portable = Ordinateur()

# print(f'classe {Ordinateur.processeur_principal}')  # Utilisation du point
# print(f'instance mon_ordi    : {mon_ordi.processeur_principal}')
# print(f'intance mon_portable : {mon_portable.processeur_principal}')


# Variable d'instance
# class Ordinateur:
#     pass


# mon_ordi = Ordinateur()
# mon_portable = Ordinateur()

# mon_ordi.marque = 'Dell'
# mon_portable.marque = 'Apple'

# print(mon_ordi.marque)
# print(mon_portable.marque)

# mon_ordi = Ordinateur()
# mon_portable = Ordinateur()

# # Variable d'instance
# class Ordinateur:
#     processeur_principal = 'CPU'

#     def __init__(self, marque):
#         self.marque = marque


# mon_ordi = Ordinateur('Dell')
# mon_portable = Ordinateur('Apple')

# print(f'instance mon_ordi    : {mon_ordi.marque}')
# print(f'intance mon_portable : {mon_portable.marque}')

# # print(f'classe {Ordinateur.marque}')  # Erreur

# # Usage du mot clé 'self' cf méthodes

# # Méthodes
# class Ordinateur:
#     processeur_principal = 'CPU'

#     def __init__(self, marque):
#         self.marque = marque

#     def demarrer(self):
#         print("minute papillon, je démarre")
#         return "Hello world"

#     # def lancer_application(self, nom):
#     #     applications = ['Word', 'Tetris']
#     #     if nom in applications:
#     #         return nom
#     #     else:
#     #         return None


# mon_ordi = Ordinateur('Dell')
# # mon_portable = Ordinateur('Apple')

# message_mon_ordi = mon_ordi.demarrer()
# print(f'instance mon_ordi    : {message_mon_ordi}')

# print('Appli lancée : ', mon_ordi.lancer_application('Tetris'))

# Usage du mot clé 'self'
# mon_portable = Ordinateur('Apple')
# mon_portable.demarrer()

# message_mon_portable = Ordinateur.demarrer(mon_portable)
# print(f'intance mon_portable : {message_mon_portable}')

# print(f'classe {Ordinateur.marque}')  # Erreur

# Usage du mot clé 'self' cf méthodes
