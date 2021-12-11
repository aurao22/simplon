"""
Boulangerie
Un simulateur fait "vivre" trois personnages économiques que sont :
1. Le Paysan
2. Le Meunier
3. Le Boulanger

Les classes à construire sont :
1. Le Paysan
. Mange du pain pour vivre . Produit du blé
. Livre du blé à un menier . Reçcoit du pain
2. Le Meunier
. Mange du pain pour vivre . Reçcoit du blé
. Produit de la farine en consommant du blé . Livre de la farine à un boulanger
. Reçcoit du pain
3. Le Boulanger
. Mange du pain (de son propre stock) pour vivre
. Reçcoit de la farine
. Fabrique du pain en consommant de la farine
. Livre du pain aux meuniers et aux paysans (en fait, à tous ceux qui peuvent en recevoir)

Etablir les trois "identités de classe" des personnages, en décrivant les attributs nécessaires.

Peut-on trouver un principe "plus abstrait" qui permette de factoriser certains éléments ?
Etablir cette classe.

Identifier les comportements communs et les résoudre dans cette classe.

Modifier et terminer les classes personnages pour tenir compte de cette factorisation.


Evaluer une manière "intéressante" de procéder aux échanges de matière.

Ecrire une petite classe "enveloppante" qui mette en scéne trois personnages et les fassent
"vivre" dans une boucle en mode "tour par tour".
"""
