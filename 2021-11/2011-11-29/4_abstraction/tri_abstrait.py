"""
L’objectif est de définir une classe abstraite destinée à gérer un tableau trié d’éléments
et comportant une méthode abstraite plusGrand(self,a,b). Cette méthode devra comparer deux éléments.
Pour gérer un tableau trié d’objets d’un certain type,
il faudra étendre la classe abstraite en une classe définissant la méthode plusGrand(self,a,b) pour le type d’objets en question.
On construira :
1. Une classe abstraite TableauTrieAbstrait gérant un tableau d’éléments
  qui reste toujours trié par ordre croissant par rapport à la relation définie par une méthode abstraite plusGrand.
  Cette classe devra contenir au moins
  - une méthode abstraite plusGrand(self,a,b)
  - une méthode inserer(self,element) pour insérer une instance d’élément dans le tableau
  (il faut faire attention à l’ordre pour ne pas l’insérer n’importe où)
2. Une classe TableauTrieEntiers qui étend la classe TableauTrieAbstrait.
   Cette classe est destinée à gérer un tableau trié d’entier.
   Il faut essentiellement y définir la méthode plusGrand(self,a,b) pour des entiers.
3. Une classe, TableauTrieChaines qui étend la classe TableauTrieAbstrait.
   Cette classe est destinée à gérer un tableau trié de chaîne de caractères.
   Il faut essentiellement y définir la méthode plusGrand(self,a,b) en se basant sur le nombre de caractères.
"""

# Version sans décorateurs


class TableauTrieAbstrait:
    """
    classe abstraite destinée à gérer un tableau d'éléments triés
    """

    def __init__(self):
        self.tableau = []  # Tableau stocké sous la forme d'une liste

    def __repr__(self):
        return str(self.tableau)

    def plus_grand(self, a, b):
        """
        Méthode abstraite
        On s'assure que toutes les classes filles implémentent cette méthode
        """
        raise NotImplementedError()

    def inserer(self, element):
        """
        Méthode commune d'insersion d'un élément dans le tableau
        On compare notre élément à toutes les valeurs du tableau par ordre croissant
        On l'insère dès que élément < valeur[tableau]
        """
        nouveau_tableau = []

        # >>> Début de votre code

        # Les premières valeurs du tableau

        # L'élément en question

        # Les dernières valeurs du tableau

        #     Fin de votre code <<<

        self.tableau = nouveau_tableau


class TableauTrieEntiers(TableauTrieAbstrait):
    def plus_grand(self, a, b):
        # >>> Début de votre code
        pass
        #     Fin de votre code <<<


class TableauTrieChaines(TableauTrieAbstrait):
    def plus_grand(self, a, b):
        # >>> Début de votre code
        pass
        #     Fin de votre code <<<


tableau_trie_d_entiers = TableauTrieEntiers()
print(tableau_trie_d_entiers)
tableau_trie_d_entiers.inserer(5)
print(tableau_trie_d_entiers)
tableau_trie_d_entiers.inserer(2)
print(tableau_trie_d_entiers)
tableau_trie_d_entiers.inserer(3)
print(tableau_trie_d_entiers)
tableau_trie_d_entiers.inserer(6)
print(tableau_trie_d_entiers)
tableau_trie_d_entiers.inserer(1)
print(tableau_trie_d_entiers)

tableau_trie_de_chaines = TableauTrieChaines()
tableau_trie_de_chaines.inserer('Lanion')
tableau_trie_de_chaines.inserer('heure')
tableau_trie_de_chaines.inserer('roupille')
tableau_trie_de_chaines.inserer('une')
tableau_trie_de_chaines.inserer('a')
print(tableau_trie_de_chaines)
