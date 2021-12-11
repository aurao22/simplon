# Boulangerie

class Humain:
    def __init__(self) -> None:
        self.produit
        self.livrable
        self.nourriture
        self.recu
        pass

    def produire(self):
        pass
    
    def livrer(self):
        pass
    
    def consomme(self):
        pass
    
    def recevoir(self):
        pass

    def getProduit(self):
        return self.produit
    def setProduit(self,produit):
        self.produit = produit
        return self.produit
    
    def getLivrable(self):
        return self.livrable
    def setLivrable(self, livrable):
        self.livrable = livrable
        return self.livrable
    
    def getNourriture(self):
        return self.nourriture
    def setNourriture(self, nourriture):
         self.nourriture = nourriture 
         return self.nourriture

    def getRecu(self):
        return self.recu
    def setRecu(self, recu):
        self.recu = recu
        return self.recu   



class Paysan(Humain):
    def __init__(self) -> None:
        super().__init__()
    """ self.recu = 'pain'
        self.nourriture = 'pain'
        self.produit = 'blé'
        self.livrable = 'blé' """

    def produire(self):
        self.__produit = 'blé'
        return self.__produit 
    
    def livrer(self): # livre le blé qu'il a produit
        self.__livrable = self.__produit
        return self.__livrable
    
    def consomme(self): # mange le pain qu'il a reçu du Boulanger
        self.__nourriture = self.__recu
        return self.__nourriture
    
    def recevoir(self, recu): # ne recoit que du Boulanger ?
        if recu is not None :
            self.__recu = recu
        return self.__recu

class Meunier(Humain):
    def __init__(self) -> None:
        super().__init__()
        """ self.recu = Paysan.getLivrable()
        self.nourriture = Boulanger.getLivrable()
        self.produit = 'farine'
        self.livrable = 'farine' """

    def produire(self, recu): #ne peut produire que si il a reçu du Paysan
        if recu is not None :
            self.__produit = 'farine'
        return self.__produit
    
    def livrer(self):
        self.__livrable = self.__produit # ne vas qu'à un Boulanger
        return self.__livrable
    
    def consomme(self): # consomme le pain reçu(?) du Boulanger
        return self.__produit
    
    def recevoir(self, recu): # recois du blé du Paysan et du Pain du Boulanger ? ou juste le blé ?
        if recu is not None :
            self.__recu = recu
        return self.__recu

class Boulanger(Humain):
    def __init__(self) -> None:
        super().__init__()
        """ self.recu = 'farine'
        self.nourriture = 'pain'
        self.produit = 'pain'
        self.livrable = 'pain' """

    def produire(self, recu): # ne peut produire que si il a reçu de Meunier
        return self.__produit
    
    def livrer(self): # livre aux autres son produit
        self.__livrable = self.__produit
        return self.__livrable
    
    def consomme(self): # consomme son propre produit
        self.__nourriture = self.__produit
        return self.__nourriture
    
    def recevoir(self, recu): # recoit de Meunier
        if recu is not None:
            self.__recu = recu
        return self.__recu    
      
paysan = Paysan()
meunier = Meunier()
boulanger = Boulanger()

# à automatiser 
paysan.produire()
paysan.livrer()
meunier.recevoir()
meunier.produire()
meunier.livrer()
boulanger.recevoir()
boulanger.produire()
boulanger.livrer()
boulanger.consomme()
paysan.recevoir()
paysan.consomme()
meunier.recevoir()
meunier.consomme()

