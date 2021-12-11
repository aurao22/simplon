"""
Gestion d'heures complémentaires
Chaque enseignant de l’université effectue un certain nombre d’heures d’enseignement dans une année.
Suivant le statut de l’enseignant, un certain nombre de ces heures peut-être considéré comme complémentaire.

Les heures complémentaires sont payées séparément à l’enseignant.
Les volumes horaires sont exprimés en heures entières et le prix d’une heure complémentaire est de 10 Euros.
Le nom et le nombre d’heures total d’un enseignant sont fixés à sa création,
puis seul le nom peut être librement consulté (méthode nom()).
D’autre part on veut pouvoir librement consulter un enseignant sur son volume d’heures complémentaires (méthode hc())
 et sur la rétribution correspondante (méthode retribution()).
Il y a deux types d’enseignants :
    les intervenants extérieurs : toutes les heures effectuées sont complémentaires,
    les enseignants de la fac : seules les heures assurées au delà d’une charge statutaire de 192h sont complémentaires.

Comment modifier le modèle pour y introduire les  étudiants de troisième cycle qui assurent des enseignements :
 toutes les heures effectuées sont complémentaires mais dans la limite de 96 heures.
"""
class Heures():

    def __init__(self, nom, nombre, prix) -> None:
        self.nom = nom
        self.nombre = nombre
        self.prix = prix

    def retribution(self):
        return self.nombre * self.prix


class HeuresComplementaire(Heures):

    # le prix d’une heure compl´ementaire est de 10 Euros
    prix_heure_complementaire = 10

    def __init__(self, nombre) -> None:
        super().__init__("Complementaire", nombre, HeuresComplementaire.prix_heure_complementaire)

class HeuresStatutaire(Heures):

    prix_heure_statuaire = 5

    def __init__(self, nombre) -> None:
        super().__init__("Statuaires", nombre, HeuresStatutaire.prix_heure_statuaire)


class Enseignants():
   
    def __init__(self, nom, heures_complementaires):
        self.nom = nom
        self.heures_complementaires = heures_complementaires

    def get_nom(self):
        return self.nom

    def hc(self):
        return self.heures_complementaires.nombre

    def retribution(self):
        return self.heures_complementaires.retribution()

    def nb_heures(self):
        return self.hc()

class IntervenantExterieur(Enseignants):

    def __init__(self, nom, nb_heures):
        super().__init__(nom, HeuresComplementaire(nb_heures))

class EnseignantsFac(Enseignants):

    # seules les heures assur´ees au del`a d’une charge statutaire de 192h sont complémentaires
    nb_heures_annuelles = 192

    def __init__(self, nom, heures):
        if heures <= EnseignantsFac.nb_heures_annuelles:
            self.heures_statuaires = HeuresStatutaire(heures)
            super().__init__(nom, HeuresComplementaire(0))
        else:
            hc = heures - EnseignantsFac.nb_heures_annuelles
            self.heures_statuaires = HeuresStatutaire(EnseignantsFac.nb_heures_annuelles)
            super().__init__(nom, HeuresComplementaire(hc))

    def retribution_statuaire(self):
        return self.heures_statuaires.retribution()

    def retribution_hc(self):
        return self.heures_statuaires.retribution()

    def retribution(self):
        return super().retribution() + self.retribution_statuaire()

    def nb_heures(self):
        return super().nb_heures() +  self.heures_statuaires.nombre       


class EtudiantsEnseignants(Enseignants):

    # les ´etudiants de troisi`eme cycle qui assurent des enseignements : toutes les heures e↵ectu´ees sont compl´ementaires mais dans la limite de 96 heures.
    max_heures = 96

    def __init__(self, nom, heures):
        if heures <= EtudiantsEnseignants.max_heures:
            super().__init__(nom, HeuresComplementaire(heures))
        else:
            super().__init__(nom, HeuresComplementaire(EtudiantsEnseignants.max_heures))


gerard = Titulaire('Gerard', 195)
print(gerard.nom(), gerard.retribution())

germaine = Intervenant('Germaine', 520)
print(germaine.nom(), germaine.retribution())

gaston = Etudiant('Gaston', 12)
print(gaston.nom(), gaston.retribution())

