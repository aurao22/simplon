-- 
-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- 
-- SGBD – SQL - TP: SELECT MONO-TABLES
-- 
-- Mettez les réponses directement dans le fichier, à la suite des questions.
-- Au mieux, vous devez faire toutes les questions.
-- Si vous allez lentement, faites d''abord les questions sans (*) des exercices 0, 1 et 2.
-- Ensuite, au choix, faites les questions avec (*) des exercices 1 et 2 ou les question des exercices exercices 3 et 4 en privilégiant d''abord celles sans (*).
-- Pour ceux qui ont le plus de facilités : concentrez-vous sur les dernières questions des exercices 2 et 3 et sur les exercices 2 et 4.
-- 
-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- EXERCICE 0:
-- 1. Démarrez le serveur.
-- 2. Démarrez un client dans le terminal.
-- 3. Affichez la version du serveur dans le client léger dans le terminal : select version();
-- 4. Affichez les BD présentes sur votre SGBD : show databases;
-- 5. Charger les tables de la base de données décrite dans le fichier EmployesTP01.sql
--    Le plus simple est de faire un copier-coller du contenu du fichier .sql dans le client léger dans le terminal.

-- Cette BD correspond à la table suivante : 

-- EMPLOYES(NE, NOM, FONCTION, DATEMB, SAL, COMM, ND)
-- •	NE	numéro de l’employé. Clé primaire.
-- •	NOM	nom de l’employé.
-- •	FONCTION	intitulé du poste occupé.
-- •	DATEMB	date d’embauche.
-- •	SAL	salaire de l’employé.
-- •	COMM	commission (part de salaire variable).
-- •	ND	n° du département dans lequel travaille l’employé.

-- 6. Affichez les BD présentes sur votre SGBD : show databases();
--    Vérifiez que la BD "employesTP01" a bien été créée.
-- 7. Selectionnez la BD "employesTP01" : use employesTP01;
-- 8. Affichez les tables de la BD "employesTP01" : show tables;
-- 9. Affichez la structure de la table "employes" : desc employes;
--    La commande desc est un diminutif de describe.
-- 10.Lisez le contenu du fichier EmployesTP01.sql. Cherchez à tout comprendre.

-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- EXERCICE 1: interrogation de la BD - Projection, restriction, distinct et tri

-- Travail à faire
-- •	Répondre à toutes les questions ci-dessous.
-- •	Le fichier répond déjà aux 2 premières questions du TP: vous devez utiliser le même formalisme pour les autres questions.
-- •	Pour chaque question vous devez mettre:
-- 	a	L’intitulé de la question
-- 	b	Le select de la réponse
-- 	c	La CP de la table résultat de chaque question
-- ****************************************************************************
-- Version réduite (si demandée)
-- •	Répondez uniquement aux questions qui ne commencent pas par (*), 
-- ****************************************************************************
-- Attention!
-- •	Vous ne devez projeter que les attributs nécessaires: CP, CS, AD, AR – Tri devant ou derrière (Clé Primaire, Clé Significative, Attributs Demandés, Attributs de restriction).
-- •	Vous devez précisez la clé primaire de chaque table de résultats.
-- ****************************************************************************
-- Outils
-- •	Utilisez un éditeur efficace: notepad++, sublimtext: qui fasse de la coloration syntaxique.
-- •	Mettez une extension .sql à vos fichiers pour avoir la coloration syntaxique.
-- •	Faites des copier-coller depuis l''éditeur jusque dans le client dans le terminal.
﻿-- employesTP01.EMPLOYES
-- Création de la BD  : on supprime la database, on la recrée, on l'utilise

drop database if exists employesTP01;
create database employesTP01;
use employesTP01;

-- Création des tables : on n'a pas besoin de droper les tables puisqu'on a commencé par droper la BD

CREATE TABLE EMPLOYES (
	NE        integer primary key auto_increment,
	NOM       varchar(10) not null,
	FONCTION  varchar(9) not null,
	DATEMB    date not null,
	SAL       float(7,2) not null,
	COMM      float(7,2),
	ND        integer not null
 );

INSERT INTO EMPLOYES VALUES (7839,'KING','PRESIDENT','1981-11-17',5000,NULL,10);
INSERT INTO EMPLOYES VALUES (7698,'BLAKE','MANAGER','1981-05-1',2850,NULL,30);
INSERT INTO EMPLOYES VALUES (7782,'CLARK','MANAGER','1981-06-9',2450,NULL,10);
INSERT INTO EMPLOYES VALUES (7566,'JONES','MANAGER','1981-04-2',2975,NULL,20);
INSERT INTO EMPLOYES VALUES (7654,'MARTIN','SALESMAN','1981-09-28',1250,1400,30);
INSERT INTO EMPLOYES VALUES (7499,'ALLEN','SALESMAN','1981-02-20',1600,300,30);
INSERT INTO EMPLOYES VALUES (7844,'TURNER','SALESMAN','1981-09-8',1500,0,30);
INSERT INTO EMPLOYES VALUES (7900,'JAMES','CLERK','1981-12-3',950,NULL,30);
INSERT INTO EMPLOYES VALUES (7521,'WARD','SALESMAN','1981-02-22',1250,500,30);
INSERT INTO EMPLOYES VALUES (7902,'FORD','ANALYST','1981-12-3',3000,NULL,20);
INSERT INTO EMPLOYES VALUES (7369,'SMITH','CLERK','1980-12-17',800,NULL,20);
INSERT INTO EMPLOYES VALUES (7788,'SCOTT','ANALYST','1982-12-09',3000,NULL,20);
INSERT INTO EMPLOYES VALUES (NULL,'ADAMS','CLERK','1983-01-12',1100,NULL,20);
INSERT INTO EMPLOYES VALUES (NULL,'MILLER','CLERK','1982-01-23',1300,NULL,10);
-- ****************************************************************************
-- Les requêtes
-- 1.	Tous les employés avec tous leurs attributs
SELECT * FROM emp;
-- 2.	Tous les employés avec leur numéro d’employé et leur nom
SELECT NE, NOM FROM emp;
-- 3.	Tous les employés triés par ordre alphabétique
SELECT * FROM emp ORDER BY NOM;
-- 4.	Tous les employés triés par n° de département croissant, ordre alphabétique des fonctions, ancienneté décroissante
SELECT * FROM emp ORDER BY ND ASC, fonction ASC, DATEMB DESC;
-- 5.	(*) Tous les employés avec leurs salaires triés par salaire décroissant
SELECT * FROM emp ORDER BY SAL DESC;
-- 6.	(*) Tous les employés du département 30 avec tous leurs attributs
SELECT * FROM emp WHERE ND = 30;
-- 7.	Tous les employés du département 30 triés par ordre alphabétique
SELECT * FROM emp WHERE ND = 30 ORDER BY NOM;
-- 8.	Tous les managers des départements 20 et 30
SELECT * FROM emp WHERE ND > 10 AND ND < 40 AND FONCTION = 'MANAGER' ORDER BY NOM;
SELECT * FROM emp WHERE ND in (20, 30) AND FONCTION = 'MANAGER';
-- 8 bis : 
    -- Tous les employés appartement à un département dont la valeur est placée dans une variable @nd.
    -- Donner 10 à @nd et lancer la requête.
    -- Donner 20 à @nd et lancer la requête.
SET @nd = 10;
SELECT * FROM employes WHERE ND = @nd;

SET @nd = 20;
SELECT * FROM employes WHERE ND = @nd;

-- 8 ter : 
    -- Créer une vue apppelée managers qui contiennent les managers avec tous leurs attributs sauf leur fonction.
CREATE VIEW employestp01.view_manager AS SELECT NE, NOM, DATEMB, SAL, COMM, ND FROM emp WHERE fonction = 'MANAGER' ;
--     Vérifier que cette vue a bien été créée
--     Ecrire une requêtes qui affiche tous les éléments de cette vue avec tous leurs attributs
SELECT * FROM view_manager;
--     Ecrire une requête qui affiche tous les managers des départements 20 et 30 en partant de cette vue.
SELECT * FROM view_manager WHERE ND = 20 OR ND = 30;
SELECT * FROM view_manager WHERE ND in (20, 30);
SELECT * from view_manager WHERE nd = (20,30);


-- 9.	(*) Tous les employés qui ne sont pas managers et qui sont embauchés en 1981
SELECT * FROM emp 
WHERE FONCTION != "MANAGER" 
AND YEAR(DATEMB) = 1981;
    
select * from emp WHERE NOT FONCTION = 'MANAGER' AND YEAR(DATEMB) =1981;

SELECT * FROM employes
WHERE fonction != "manager" AND EXTRACT(YEAR FROM datemb) = 1981;

SELECT * FROM EMPLOYES WHERE FONCTION != 'MANAGER'AND DATEMB LIKE '1981%';
-- 9 bis : 
--     Tous les employés embauché une certaine année. La valeur de l''année est placée dans une variable @annee.
--     Donner 1981 à @nd et lancer la requête.
--     Donner 1980 à @nd et lancer la requête.
SET @annee = 1981;
SELECT * FROM emp WHERE YEAR(DATEMB) = @annee;
-- 10.	Toutes les fonctions de la société
SELECT DISTINCT(FONCTION) FROM emp;

-- 11.	Tous les employés ne travaillant pas dans le département 30 
-- et qui soit ont un salaire > à 2800, soit sont manager.
SELECT * FROM emp
WHERE ND != 30
AND (SAL > 2800 
OR FONCTION = "MANAGER");
-- 12.	(*) Tous les employés dont le salaire est compris entre 1000 et 2000
SELECT * FROM emp
WHERE ND != 30
AND SAL > 1000 AND SAL < 2000;
-- 13.	Tous les numéros de département
SELECT DISTINCT(ND)
FROM emp;
-- 14.	Toutes les fonctions par département (10: président, 10: manager, etc.)
SELECT ND,FONCTION
FROM emp
GROUP BY ND,FONCTION;
-- 15.	Tous les employés ayant ou pouvant avoir une commission
SELECT *
FROM emp
WHERE COMM IS NOT NULL;
-- 16.	(*) Tous les salaires, commissions et totaux (salaire + commission) des vendeurs
SELECT SAL, COMM, SAL + COMM as TOT
FROM emp
WHERE FONCTION="SALESMAN";
-- 17.	Tous les salaires, commissions et totaux (salaire + commission) des employés
SELECT SAL, COMM, IFNULL(SAL, 0) + IFNULL(COMM, 0) as TOT
FROM emp;
-- 18.	(*) Tous les employés embauchés en 1981
SELECT * FROM emp WHERE YEAR(DATEMB) = 1981;
-- 19.	Tous les employés avec leur date d’embauche, la date du jour 
-- et leur nombre d’années d’ancienneté (on considérera que toute année commencée vaut pour une année), 
-- triés par ancienneté
-- (on utilisera les fonctions de base de traitement de date et de traitement de chaîne).
SELECT *, YEAR(CURDATE()) - YEAR(DATEMB) as ANCIENNETE
FROM emp
ORDER BY ANCIENNETE;

SELECT *, CURDATE(), timestampdiff(YEAR, DATEMB, CURDATE()) as ANCIENNETE
FROM emp
ORDER BY ANCIENNETE;

-- 20.	(*) Tous les employés ayant un A en troisième lettre de leurs noms.
SELECT NOM
FROM emp
WHERE NOM LIKE "__A%";

SELECT NOM
FROM emp
WHERE POSITION("A" IN NOM)=3;

-- 21.	Tous les employés ayant au moins deux A dans leurs noms.
SELECT NOM
FROM emp
WHERE NOM LIKE "%A%A%";

SELECT NOM
FROM emp
WHERE ROUND(
        (
            LENGTH(NOM)
            - LENGTH( REPLACE ( NOM, "A", "") ) 
        ) / LENGTH("A")        
    ) > 1;

-- 22.	(*) Donner les quatre dernières lettres du nom de chaque employé.
select NOM, SUBSTRING(NOM, -4) from emp;

-- 23.	Quel est le plus gros salaire de l’entreprise? FONCTION MAX interdite.
select MAX(SAL)from emp;
select SAL from emp ORDER BY SAL desc LIMIT 1;
SELECT SAL, COMM, IFNULL(SAL, 0) + IFNULL(COMM, 0) as TOT
FROM emp
ORDER BY TOT desc LIMIT 1;
-- 24.	(*) Quel est le plus gros salaire total des vendeurs (SALESMAN)? FONCTION MAX interdite.
SELECT SAL, COMM, IFNULL(SAL, 0) + IFNULL(COMM, 0) as TOT
FROM emp
WHERE FONCTION="SALESMAN"
ORDER BY TOT desc LIMIT 1;
-- 25.	Lister 3 employés au hasard
SELECT NOM
  FROM emp AS r1 JOIN
       (SELECT CEIL(RAND() *
                     (SELECT MAX(NE)
                        FROM emp)) AS id)
        AS r2
 WHERE r1.NE >= r2.id
 ORDER BY r1.NE ASC
 LIMIT 3

-- 26.	(*) Afficher tous les employés en affichant: 
-- «anciens» pour ceux embauchés avant le 1 janvier 1982, rien pour ceux embauchés en 1982 et 
-- «nouveaux» pour ceux embauchés après le 1 janvier 1983. 
-- On utilisera deux méthodes: le case et le if. On tri par date d’embauche et par ordre alphabétique.

SELECT NOM, DATEMB,
CASE
    WHEN YEAR(DATEMB) > 1982 THEN "nouveaux"
    WHEN YEAR(DATEMB) < 1982 THEN "anciens"
    ELSE ""
END  as anciennete
FROM emp
ORDER BY NOM, DATEMB; 

SELECT NOM, DATEMB, IF(YEAR(DATEMB) > 1982, "nouveaux", IF(YEAR(DATEMB) < 1982, "ancien", "")) as anciennete
FROM emp
ORDER BY NOM, DATEMB; 

-- 27.	Afficher les employés avec le numéro de leur tranche de salaire. Le numéro va de 1 à N. 
-- La première tranche va de 0 à 999, la deuxième de 1000 à 1999, la troisième de 2000 à 2999, etc. 
-- On considère qu’on ne sait pas à l’avance combien il y aura de tranche. 
-- On affichera les résultats par ordre de tranche croissante et par ordre alphabétique à l’intérieur d’une même tranche.
-- On affiche tous les attributs et la tranche à côté du salaire.

SELECT ROUND(MAX(SAL)/1000) as nb_tranches FROM emp;

SELECT @nb_tranches := ROUND(MAX(SAL)/1000) FROM emp;
SET @inc = 0;

SELECT NOM, SAL,
WHILE inc < nb_tranches DO
    SET inc = inc + 1;
    CASE
		WHEN ROUND(SAL/1000) = inc THEN CONCAT("Tranche ", inc, " € to ", inc + 1000, " €")
	END
END as tranche
FROM emp
ORDER BY NOM, SAL; 

SELECT *, IFNULL(SAL, 0) + IFNULL(COMM, 0) as TOT, CEILING((IFNULL(SAL, 0) + IFNULL(COMM, 0)) / 999) as Tranche
FROM emp
ORDER BY Tranche, TOT, SAL, NOM; 

-- 27 bis : 
--     Créer une vue apppelée tranches qui contiennent la requête précédente.
--     Vérifier que cette vue a bien été créée
--     Ecrire une requêtes qui affiche tous les éléments de cette vue avec tous leurs attributs.
--     Ecrire une requête qui affiche les salariés dont la tranche est >3 avec leur salaire.
CREATE VIEW view_tranches AS 
SELECT *, IFNULL(SAL, 0) + IFNULL(COMM, 0) as TOT, CEILING((IFNULL(SAL, 0) + IFNULL(COMM, 0)) / 999) as Tranche
FROM emp
ORDER BY Tranche, TOT, SAL, NOM; 

SELECT *
FROM view_tranches
WHERE Tranche > 3;

-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- EXERCICE 2: interrogation de la BD

-- Travail à faire
-- La même chose que pour l’exercice 1, à la suite dans le même fichier.
-- ****************************************************************************
-- Les requêtes
-- 1.	Combien d''employés dans la société
SELECT count(NE)
FROM emp;

-- 2.	Combien d''employés embauchés en 1981
SELECT count(NE)
FROM emp
WHERE YEAR(DATEMB) = 1981;

-- 3.	Combien de vendeurs («Salesman») dans la société
SELECT count(NE)
FROM emp
WHERE FONCTION = "SALESMAN";

-- 4.	Combien de départements dans la société
SELECT count(DISTINCT(ND))
FROM emp;

-- 5.	Combien de fonctions différentes dans la société
SELECT count(DISTINCT(FONCTION))
FROM emp;

-- 6.	(*) Combien y a-t-il d’employés qui n’ont pas et ne peuvent pas avoir de commission?
SELECT count(NE)
FROM emp
WHERE COMM is NULL;

-- 7.	(*) Salaires minimum, maximum et moyen de la société.
SELECT MIN(SAL), MAX(SAL), avg(SAL)
FROM emp;

-- 8.	Salaires moyens des vendeurs
SELECT MIN(SAL), MAX(SAL), avg(SAL)
FROM emp
WHERE FONCTION = "SALESMAN";

-- 9.	Salaires moyens de tous les employés en tenant compte des commissions
SELECT MIN(TOT), MAX(TOT), avg(TOT), MIN(SAL), MAX(SAL), avg(SAL)
FROM view_tranches;

-- 10.	(*) Pourcentage moyen de la commission des vendeurs par rapport à leur salaire
SELECT SAL, COMM, TOT, ROUND((COMM / TOT) * 100) as poucentage_commission
FROM view_tranches
WHERE COMM is not NULL;

-- 11.	(*) Quel est le salaire moyen, les salaires min et max et le nombre d’employé par profession?
SELECT FONCTION, count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL)
FROM emp
GROUP BY FONCTION
ORDER BY nb_sal;

-- 12.	Quels sont les salaires maximums de chaque département?
SELECT ND, count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL)
FROM emp
GROUP BY ND
ORDER BY nb_sal;

-- 13.	(*) Quels sont les départements dans lesquels travaillent plus de deux personnes et 
-- quels sont les salaires moyens dans ces départements?
SELECT ND, count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL)
FROM emp
GROUP BY ND
HAVING nb_sal > 2
ORDER BY nb_sal;

-- 14.	Quels sont les départements dans lequel il y a plus que 4 personnes?
SELECT ND, count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL)
FROM emp
GROUP BY ND
HAVING nb_sal > 4
ORDER BY nb_sal;

-- 15.	Quels sont les fonctions pour lesquels la moyenne du salaire est supérieure à 2000?
SELECT FONCTION, count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL) as sal_moy
FROM emp
GROUP BY FONCTION
HAVING sal_moy > 2000
ORDER BY nb_sal;

-- 16.	Combien y a-t-il d’employés par département et par fonction et quelle est la moyenne de leurs salaires?
SELECT ND, FONCTION, count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL) as sal_moy
FROM emp
GROUP BY ND, FONCTION
ORDER BY nb_sal;

-- 17.	(*) Quel est le nombre d’employés par année d’embauche?
SELECT YEAR(DATEMB), ND, FONCTION, count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL) as sal_moy
FROM emp
GROUP BY YEAR(DATEMB)
ORDER BY nb_sal;

-- 18.	Combien y a-t-il d’employés par tranches de salaire de 1000 (0 à 999, 1000 à 1999, etc.).
SELECT count(NE) as nb_sal, MIN(SAL), MAX(SAL), avg(SAL) as sal_moy
FROM view_tranches
GROUP BY Tranche
ORDER BY nb_sal;

-- 19.	(*) Combien d’employés et de départements par fonction. Vérifiez bien votre résultat :  
-- par exemple, il y a deux ANALYST, tous les deux dans le département 20, la réponse est donc: 2 employés 
-- et 1 département pour la fonction ANALYST.
SELECT FONCTION, count(distinct(ND)) as nb_dep, count(distinct(NE)) as nb_sal
FROM emp
GROUP BY FONCTION;

-- 20.	Liste des employés par fonction.
SELECT *
FROM emp
ORDER BY FONCTION;

-- 21.	(*) Liste des employés par départements.
SELECT *
FROM emp
ORDER BY ND;

-- 22.	Liste des fonctions par département.
SELECT ND, FONCTION
FROM emp
GROUP BY ND;

-- 23.	(*) Liste des départements par fonction avec le nombre d''employés par fonction.
SELECT FONCTION, ND, count(NE)
FROM emp
GROUP BY FONCTION;

-- 24.	Quel est le nombre d’employés par département et par fonction, trié par département décroissant.
SELECT ND, FONCTION, count(NE)
FROM emp
GROUP BY ND, FONCTION
ORDER BY ND;

-- 25.	La même chose, mais trié par nombre d’employé et par numéro de département croissant. On fera une version avec alias.
SELECT ND, FONCTION, count(NE) as nb_sal
FROM emp
GROUP BY ND, FONCTION
ORDER BY nb_sal, ND;

-- 26.	Combien y a-t-il d’employés et de départements par fonction. 
-- Vérifiez bien votre résultat :  par exemple, il y a deux ANALYST, tous les deux dans le département 20, 
-- la réponse est donc: 2 employés et 1 département pour la fonction ANALYST.
SELECT FONCTION, count(distinct(ND)) as nb_dep, count(distinct(NE)) as nb_sal
FROM emp
GROUP BY FONCTION;

-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- EXERCICE 3: BD Cinema - Projection, restriction, distinct et tri

-- Travail à faire
-- La même chose que pour les exercices précédents, à la suite dans le même fichier.
-- ****************************************************************************
-- 1. Chargez la BD cinéma : copier le contenu du script dans une calculette.
-- 2. Il y a des erreurs : lisez bien le message. Il précise où sont les erreurs.
--    Corrigez les erreurs.
-- 3. Lisez le script de création de la BD. Cherchez à tout comprendre.
-- 4. Affichez tous les films avec tous les attributs
-- 5. (*) Affichez tous les films triés par ordre alphabétique.
-- 6. Affichez tous les films de Abbas Kiarostami classé par année de sortie.
-- 7. Affichez tous les films de Mankiewicz classé par année de sortie.
-- 8. Affichez tous les films d''avant 1940, trié par année et par ordre alphabétique de réalisateur.
-- 9. (*) Affichez tous les films dont la note de CJP (coteCJP) est différente de 0. Classez les par note décroissante.
-- 10.Affichez tous les films des années 1950, 
--       trié par noteCJP décroissante (48 est la meilleure note) 
--       et noteSK croissante (1 est la meilleure note) et par année.
-- 11.Affichez les realisateurs des années 1950 triés par ordre alphabétique.

-- ****************************************************************************
-- ****************************************************************************
-- ****************************************************************************
-- EXERCICE 4: BD Cinema

-- Travail à faire
-- La même chose que pour l’exercice 3, à la suite dans le même fichier.
-- ****************************************************************************
-- Les requêtes
-- 1. Affichez le nombre de films par réalisateur trié par réalisateur.
-- 5. Affichez le nombre de films par réalisateur trié par nombre de films décroissant et par réalisateur, pour les réalisateurs ayant plus de 5 films dans la BD.
-- 4. Affichez le nombre de films par annee trié par nombre de films décroissant et par année décroissante.
-- 4. Affichez le nombre de films par décennie trié par décennie décroissante.
-- 5. Combien y a-t-il de films avec une note CJP > 0 ? Idem pour SK et pour BL.

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
-- Cela se fait par exemple avec la commande:
-- SET character_set_client = utf8;

-- http://dev.mysql.com/doc/refman/5.7/en/set-character-set.html
-- http://dev.mysql.com/doc/refman/5.7/en/charset-connection.html

-- Utilisation des commandes MySQL dans le client léger dans le terminal:
-- https://dev.mysql.com/doc/refman/5.7/en/mysql-commands.html
-- https://dev.mysql.com/doc/refman/8.0/en/mysql-commands.html
