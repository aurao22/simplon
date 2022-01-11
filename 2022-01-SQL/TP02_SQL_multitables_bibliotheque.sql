-- *************************************************************************
-- Votre NOM  : RAOUL
-- Votre Prénom : AURELIE
-- *************************************************************************
-- Vous devez répondre aux questions dans ce document 
-- directement sous chaque question.
-- Vous devez renvoyer ce document par mail à la fin de la séance. 
use biblioTP02;
-- CARACTERISTIQUES du MAIL obligatoires : 

-- - Objet: ESEM-4-J3-Bibliotheque-Nom-Prénom
-- - En pièce jointe : le fichier. Si le .sql pose un problème, faites un .txt
-- - adresse d''envoi : liaudet.bertrand@wanadoo.fr

-- Attention: les mails sont filtrés sur la base de l''intitulé de l''objet.
-- Ne vous trompez pas, sinon votre mail sera perdu.

-- -- Fichier texte encodé en UTF8 with BOM

-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************

-- SGBD – SQL - TP : SELECT MULTI-TABLES

-- Mettez les réponses directement dans le fichier, à la suite des questions.
-- Faites d''abords les questions sans (*) des exercices 0, 1 et 2.
-- Ensuite, au choix, faites les questiont avec (*) ou les exercices 3 et 4.

-- *********************************************************************************
-- *********************************************************************************
-- *********************************************************************************
-- EXERCICE 0 : 
-- SERIE 0 : La bibliothèque - Graphe des tables

-- 1.	Faites le graphe des tables, de façon arborescente (flèches vers le bas)

-- *******************************************************************************************
-- *******************************************************************************************
-- *******************************************************************************************
-- SERIE 1 : La bibliothèque - Consultation

-- 1.	Faites un select pour chaque table.
-- 	Combien y a-t-il de tuples dans chaque tables (regardez juste le résultat du select)
SELECT COUNT(NA) FROM adherents;
-- 33
SELECT COUNT(*) FROM categories;
-- 12
SELECT COUNT(*) FROM emprunter;
-- 45
SELECT COUNT(*) FROM livres;
-- 50

SELECT COUNT(*) FROM oeuvres;
-- 35

SELECT COUNT(*) FROM thematique;
-- 74

-- 2.	Faites la jointure naturelle de la table des Livres avec la table des Oeuvres. 
SELECT nl, o.no, titre, auteur 
FROM livres l
JOIN oeuvres o on o.no = l.no
ORDER BY titre;


-- 	On projette nl, titre, auteur et editeur.
-- 	a•	Quelle est la clé primaire de la la table résultat ?
-- nl, o.no
-- 	b.	Combien y a-t-il de tuples dans la table résultat ? Quelle formule donne le résultat ?
-- 50
-- 	c.  Créer la vue de la tables des livres avec les oeuvres : vue exemplaires
CREATE VIEW view_exemplaires AS 
SELECT nl, o.no, titre, auteur 
FROM livres l
JOIN oeuvres o on o.no = l.no
ORDER BY titre;

-- 3.	(*) Faites le produit cartésien de la table Emprunter avec la table des Livres et la table des Adhérents. 
-- 	On projettera tous les attributs.
SELECT l.nl, e.nl, a.na, e.na, nom, prenom, adr, tel, datEmp, dureeMax, dateRet
FROM emprunter e, livres l, adherents a
ORDER BY nom, datEmp;
-- 	a.	Comptez manuellement le nombre d’attributs dans la table résultat ? Quelle formule donne le résultat ?
-- 1000
-- 	b.	Combien y a-t-il de tuples dans la table résultat ? Quelle formule donne le résultat ?
-- 1000
-- 	c•	Quelle est la clé primaire de la la table résultat ?
-- l.nl, e.nl, a.na, e.na

-- 4.	(*) Faites la jointure naturelle de la table Emprunter avec la table des Livres et la table des Adhérents. 
SELECT l.nl, a.na, nom, prenom, adr, tel, datEmp, dureeMax, dateRet
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN adherents a on a.na = e.na
ORDER BY nom, datEmp;

-- 	On projettera tous les attributs utiles.
-- 	a.	Comptez manuellement le nombre de tuples dans la table résultat ? 
-- 45
-- 	b•	Quelle est la clé primaire ?
-- nl, na
-- 	c•	Triez les résultats par date d''emprunt	
SELECT l.nl, a.na, nom, prenom, adr, tel, datEmp, dureeMax, dateRet
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN adherents a on a.na = e.na
ORDER BY datEmp;
-- 	d•	Triez les résultats par adhérents et date d''emprunt
SELECT l.nl, a.na, nom, prenom, adr, tel, datEmp, dureeMax, dateRet
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN adherents a on a.na = e.na
ORDER BY nom, datEmp;

-- 5.	Faites la jointure naturelle de la table Emprunter avec la table des Livres, la table des Oeuvres et la table des Adhérents. 
-- 	On projettera tous les attributs utiles.
SELECT a.na, nom, prenom, adr, tel, l.nl, titre, auteur, datEmp, dureeMax, dateRet
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN oeuvres o on o.no = l.no
JOIN adherents a on a.na = e.na
ORDER BY nom, datEmp;

-- 	a.	Comptez manuellement le nombre de tuples dans la table résultat ? 
-- 45
-- 	b•	Quelle est la clé primaire ?
-- a.na, l.nl
-- 	c•	Triez les résultats par date d''emprunt. Interprétez le résultat.
SELECT a.na, nom, prenom, adr, tel, l.nl, titre, auteur, datEmp, dureeMax, dateRet
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN oeuvres o on o.no = l.no
JOIN adherents a on a.na = e.na
ORDER BY datEmp;

-- 	d•	Triez les résultats par adhérents et date d''emprunt. Interprétez le résultat.
SELECT a.na, nom, prenom, adr, tel, l.nl, titre, auteur, datEmp, dureeMax, dateRet
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN oeuvres o on o.no = l.no
JOIN adherents a on a.na = e.na
ORDER BY nom, datEmp;

-- 	e•	Triez les résultats par auteur et oeuvres et date d''emprunt. Interprétez le résultat.
SELECT auteur, titre, datEmp, a.na, nom, prenom, adr, tel, l.nl
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN oeuvres o on o.no = l.no
JOIN adherents a on a.na = e.na
ORDER BY auteur, titre, datEmp;

-- 	f.  Afficher le nombre d''emprunts par auteur.
SELECT auteur, count(datEmp) as nb_emprunts
FROM emprunter e
JOIN livres l on l.nl = e.nl
JOIN oeuvres o on o.no = l.no
GROUP BY auteur
ORDER BY nb_emprunts desc;


-- 6.  Creez une vue permettant d''obtenir les livres avec les informations d''auteurs.
-- 	On appellera la vue "livres_complets"
CREATE VIEW view_livres_complets AS 
SELECT nl, editeur, titre, auteur 
FROM livres l
JOIN oeuvres o on o.no = l.no
ORDER BY titre;

-- 7.  Refaites les requête 5.e et 5.f avec la vue livres_complets de l''exercice 6.
SELECT auteur, titre, datEmp, a.na, nom, prenom, adr, tel, l.nl
FROM emprunter e
JOIN view_livres_complets l on l.nl = e.nl
JOIN adherents a on a.na = e.na
ORDER BY auteur, titre, datEmp;

SELECT auteur, titre, datEmp, a.na, nom, prenom, adr, tel, l.nl
FROM emprunter e
JOIN view_livres_complets l on l.nl = e.nl
JOIN adherents a on a.na = e.na
ORDER BY auteur, titre, datEmp;

SELECT auteur, count(datEmp) as nb_emprunts
FROM emprunter e
JOIN view_livres_complets l on l.nl = e.nl
GROUP BY auteur
ORDER BY nb_emprunts desc;

-- *******************************************************************************************
-- *******************************************************************************************
-- *******************************************************************************************
-- SERIE 2 : La bibliothèque - Requêtes

-- 1.	Quels sont les livres actuellement empruntés ?
SELECT auteur, titre, datEmp, a.na, nom, prenom
FROM emprunter e
JOIN view_livres_complets l on l.nl = e.nl
JOIN adherents a on a.na = e.na
WHERE e.dateRet is NULL
ORDER BY auteur, titre, datEmp;

-- 2.	Quels sont les livres empruntés par Jeannette Lecoeur ? Vérifier dans la réponse qu’il n’y a pas d’homonymes.
SELECT auteur, titre, datEmp, a.na, nom, prenom
FROM emprunter e
JOIN view_livres_complets l on l.nl = e.nl
JOIN adherents a on a.na = e.na
WHERE a.nom like "%Lecoeur%" AND a.prenom like "%Jeanette%"
ORDER BY auteur, titre, datEmp;

-- 3.	Quels sont tous les livres empruntés le mois dernier ?
SELECT datEmp, auteur, titre
FROM emprunter e
JOIN view_livres_complets l on l.nl = e.nl
WHERE datEmp between "2021-11-30" AND "2021-12-31"
ORDER BY datEmp, auteur, titre;

-- 4.	Tous les adhérents qui ont emprunté un livre de Fedor Dostoievski.
SELECT a.na, nom, prenom
FROM adherents a
JOIN emprunter e on a.na = e.na
JOIN view_livres_complets l on l.nl = e.nl
WHERE auteur like "%Dostoievski%"
ORDER BY nom;

-- 5.	Quels sont le ou les auteurs du titre « Voyage au bout de la nuit »
SELECT distinct(auteur)
FROM view_livres_complets
WHERE titre like "%Voyage au bout de la nuit%"
ORDER BY auteur;

-- 6.	Quels sont les ou les éditeurs du titre « Narcisse et Goldmund »
select distinct(editeur) 
from view_livres_complets
where titre like "%Narcisse et Goldmund%";

-- 7.	Quels sont les adhérents actuellement en retard ?
select distinct(a.na), nom, prenom
from adherents a
INNER JOIN emprunter e on a.na = e.na
Where dateRet is NULL
AND (dureeMax < DATEDIFF(current_date(),datEmp))
ORDER BY nom, prenom;

-- 8.	Quels sont les livres actuellement en retard ?
select distinct(l.nl), titre, auteur, editeur
from view_livres_complets l
INNER JOIN emprunter e on l.nl = e.nl
Where dateRet is NULL
AND (dureeMax < DATEDIFF(current_date(),datEmp))
ORDER BY titre, editeur;

-- 9.	Quels sont les adhérents en retard avec le nombre de livre en retard et la moyenne du nombre de jour de retard.
select a.na, nom, prenom, count(e.nl) as nb_livres, round(avg(DATEDIFF(current_date(),datEmp) - dureeMax), 0) as retard_moyen
from adherents a
INNER JOIN emprunter e on a.na = e.na
Where dateRet is NULL
AND (dureeMax < DATEDIFF(current_date(),datEmp))
GROUP BY nom, prenom
ORDER BY retard_moyen, nb_livres, nom, prenom;

-- 10.	Nombre de livres empruntées par auteur.
select auteur, count(l.nl) as nb_livres_empruntes
from view_livres_complets l
INNER JOIN emprunter e on l.nl = e.nl
GROUP BY auteur
ORDER BY nb_livres_empruntes, auteur;

-- 11.	Nombre de livres empruntés par éditeur.
select editeur, count(l.nl) as nb_livres_empruntes
from view_livres_complets l
INNER JOIN emprunter e on l.nl = e.nl
GROUP BY editeur
ORDER BY nb_livres_empruntes, editeur;

-- 11. bis : Nombre de livres empruntés pas auteur et editeur.
select auteur, editeur, count(l.nl) as nb_livres_empruntes
from view_livres_complets l
INNER JOIN emprunter e on l.nl = e.nl
GROUP BY auteur, editeur
ORDER BY auteur, editeur, nb_livres_empruntes;

-- 11. ter : with rollup : Nombre de livres empruntés pas auteur et editeur
select auteur, editeur, count(l.nl) as nb_livres_empruntes
from view_livres_complets l
INNER JOIN emprunter e on l.nl = e.nl
GROUP BY auteur, editeur WITH ROLLUP
ORDER BY auteur, editeur, nb_livres_empruntes;

-- 12. Nombre de livres empruntées par auteur et par éditeur.
-- 	Ajouter ensuite un with-rollup à la fin de la requête.

-- 12.	Quelle est la durée des emprunts rendus ?
select e.nl, e.na, round(DATEDIFF(dateRet,datEmp), 0) as duree_emprunt
from emprunter e
where dateRet is not NULL
ORDER BY duree_emprunt;

-- 13. Quelle est la durée moyenne des emprunts rendus.
select avg(round(DATEDIFF(dateRet,datEmp), 0)) as duree_emprunt_moyen
from emprunter e
where dateRet is not NULL;

-- 14.	Quelle est la durée moyenne des retards pour l’ensemble des emprunts.
select round(avg(DATEDIFF(current_date(),datEmp) - dureeMax), 0) as retard_moyen
from emprunter
Where dateRet is NULL
AND (dureeMax < DATEDIFF(current_date(),datEmp));

select * from emprunter;

-- La fonction « if » permet de tester une valeur et de renvoyer ce qu’on souhaite selon la vérité ou la fausseté de la valeur testée.
-- if(a<0, 0, a) : permet de ramener les valeurs négatives de a à 0.
-- 15.	Durée moyenne des retards parmi les seuls retardataires.
-- IF(YEAR(DATEMB) > 1982, "nouveaux", IF(YEAR(DATEMB) < 1982, "ancien", "")) as anciennete
select if(dureeMax < DATEDIFF(current_date(),datEmp), DATEDIFF(current_date(),datEmp),0) as retard
from emprunter
Where dateRet is null;

select round(avg(if(dateRet is null, DATEDIFF(current_date(),datEmp),DATEDIFF(dateRet,datEmp))),0) as retard_moyen
from emprunter
Where (dateRet is not null and (dureeMax < DATEDIFF(dateRet,datEmp)));

select round(avg(if(dureeMax < DATEDIFF(current_date(),datEmp), DATEDIFF(current_date(),datEmp),0)),0) as retard_moyen
from emprunter
Where dateRet is null
AND dureeMax < DATEDIFF(current_date(),datEmp);

-- Ajouter les personnes qui ont rendu et qui était en retard
select round(avg(if(dateRet is null, DATEDIFF(current_date(),datEmp),DATEDIFF(dateRet,datEmp))),0) as retard_moyen
from emprunter
Where (dateRet is null AND dureeMax < DATEDIFF(current_date(),datEmp))
OR (dateRet is not null AND dureeMax < DATEDIFF(dateRet,datEmp));


-- *******************************************************************************************
-- *******************************************************************************************
-- *******************************************************************************************
-- SERIE 4 : La bibliothèque - Requêtes avec Thématiques

-- 1.	Quels sont les livres "jeunesse" actuellement empruntés ?

-- 2.	Afficher les catégories des livres actuellement empruntés.
-- 3.	Afficher la liste des catégories par oeuvre.
-- 4.	Quels sont les adhérents ayant empruntés des livres "jeunesse" ?
-- 5.	Combien de titres "jeunesse" ont été empruntés par adhérents ?
-- 	Commencer par lister les emprunts "jeunesse". Attention à compter les titres et pas les emprunts.
-- 6.	Dans la requete précédente, afficher en plus la liste des titres empruntés.
-- 7.	Afficher le nombre d''emprunts par catégorie.
-- 8.	Afficher la durée moyenne des emprunts par catégorie.

-- *******************************************************************************************
-- *******************************************************************************************
-- *******************************************************************************************
-- SERIE 5 : La bibliothèque - Jointures externes et/ou requêtes imbriquées

-- 1.	Quels sont les adhérents qui n’ont jamais emprunté de livres ?
-- 2.	Quels sont les livres qui n’ont jamais été empruntés ?
-- 3.	Quels sont les oeuvres qui n’ont jamais été empruntés ?
-- 3bis.	Quels sont les livres qui n’ont jamais été empruntés : afficher l''oeuvre
-- 4.	Combien d’exemplaires du titre : « Narcisse et Goldmund » sont disponibles ?
-- 	a) On commencera par afficher tous les Narcisse.
-- 	b) Puis tous les Narcisses actuellement empruntés
-- 	c) Puis les Narcisses disponibles avec une requête imbriquée
-- 	d) Enfin les Narcisse disponibles avec une jointure externes

-- *******************************************************************************************
-- *******************************************************************************************
-- *******************************************************************************************
-- SERIE 6 : La bibliothèque - Requêtes avancées

-- 1.	Pour le titre « Narcisse et Goldmund » afficher dans une même requête : 
-- 		- le nombre d’exemplaires total, 
-- 		- le nombre d’exemplaires disponibles 
-- 		- et le nombre d’exemplaires actuellement empruntés.
-- 2.	Quelle est la moyenne du nombre de livres empruntés par adhérent.
-- 3.	Refaire la question 2 pour tous les titres actuellement sortis au moins une fois.


-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- Remarque concernant la gestion des accents avec MySQL : 
-- Si on a des problèmes d’accents, faire une show variables like '%char%';
-- Sous Windows : 
-- mysql> show variables like '%char%';
-- +--------------------------+-------------------------------------+
-- | Variable_name            | Value                               |
-- +--------------------------+-------------------------------------+
-- | character_set_client     | cp850                               |
-- | character_set_connection | cp850                               |
-- | character_set_database   | latin1                              |
-- | character_set_filesystem | binary                              |
-- | character_set_results    | cp850                               |
-- | character_set_server     | latin1                              |
-- | character_set_system     | utf8                                |
-- | character_sets_dir       |
--                  c:\wamp64\bin\mysql\mysql5.7.14\share\charsets\ |
-- +--------------------------+-------------------------------------------------+
-- 8 rows in set, 1 warning (0.00 sec)
-- Sous Mac
-- mysql> show variables like '%char%';
-- +--------------------------+-------------------------------------+
-- | Variable_name            | Value                               |
-- +--------------------------+-------------------------------------+
-- | character_set_client     | utf8                                |
-- | character_set_connection | utf8                                |
-- | character_set_database   | latin1                              |
-- | character_set_filesystem | binary                              |
-- | character_set_results    | utf8                                |
-- | character_set_server     | latin1                              |
-- | character_set_system     | utf8                                |
-- | character_sets_dir       |
--                       /Applications/MAMP/Library/share/charsets/ |
-- +--------------------------+-------------------------------------+
-- 8 rows in set (0,00 sec)

-- Remplacement du character_set : 

-- Il se peut qu’il faille remplacer les character_set_client cp850 par de l’utf8.
-- Cela se fait par exemple avec la commande :
-- SET character_set_client = utf8;

-- http://dev.mysql.com/doc/refman/5.7/en/set-character-set.html
-- http://dev.mysql.com/doc/refman/5.7/en/charset-connection.html

-- Utilisation des commandes MySQL dans la calculette : 
-- https://dev.mysql.com/doc/refman/5.7/en/mysql-commands.html
-- https://dev.mysql.com/doc/refman/8.0/en/mysql-commands.html
