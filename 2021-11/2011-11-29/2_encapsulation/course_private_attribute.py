"""
Contrôle d'accès
https://towardsdatascience.com/5-ways-to-control-attributes-in-python-an-example-led-guide-2f5c9b8b1fb0

https://www.freecodecamp.org/news/python-property-decorator/


Usage d'attributs privés en Python
https://stackoverflow.com/questions/4555932/public-or-private-attribute-in-python-what-is-the-best-way

Usage de méthodes privées
https://stackoverflow.com/questions/15047122/python-private-function-coding-convention

Convention de nommage, mais restent accessibles
"""


class Test:
    def __init__(self, la_valeur_publique, la_valeur_privee):
        self.valeur_publique = la_valeur_publique
        self._valeur_privee = la_valeur_privee  # Attibut privé

    def methode_publique(self):
        pass

    def _methode_privee(self):  # Méthode privée
        pass

    def methode(self):
        self._methode_privee()


# ####### Programme principal ########
t = Test(1, 2)
print(t.valeur_publique)
print(t._valeur_privee)  # On ne le fait pas par convention
t.methode_publique()
t._methode_privee()   # On ne le fait pas par convention
t.methode()


# Décorateur @property
# https://www.freecodecamp.org/news/python-property-decorator/
class House:

    def __init__(self, price):
        self._price = price

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, new_price):
        if new_price > 0 and isinstance(new_price, float):
            self._price = new_price
        else:
            print("Please enter a valid price")

    @price.deleter
    def price(self):
        del self._price


house = House(50000.0)
print(house.price)  # getter

house.price = 45000.0   # Update value via setter
print(house.price)  # getter
# print(house._price)  # On ne le fait pas par convention
