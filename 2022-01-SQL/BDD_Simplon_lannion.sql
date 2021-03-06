drop database if exists simplon_lannion;

CREATE DATABASE simplon_lannion
    DEFAULT CHARACTER SET = 'utf8mb4';

use simplon_lannion;

CREATE TABLE ENTREPRISE (
id integer primary key auto_increment,
nom varchar(25) not NULL,
departement int DEFAULT 22,
ville varchar(25)
);

INSERT INTO ENTREPRISE VALUES
(NULL,'Event Factory',22, "BEGARD"),
(NULL,'Holai',22, "PLOUFRAGRAN"),
(NULL,'Crédit Agricole',22, "PLOUFRAGRAN"),
(NULL,'Orange',22, "LANNION"),
(NULL,'NOKIA',22, "LANNION"),
(NULL,'SAOTI',22, "LANNION"),
(NULL,'VOXYGEN',22, "PLEUMEUR-BODOU"),
(NULL,'ECO-COMPTEUR',22, "LANNION"),
(NULL,'SF2I',22, "NOUMEA"),
(NULL,'Labocéa',22, "PLOUFRAGAN"),
(NULL,'Cogédis',29, "SAINT-THONAN"),
(NULL,'Enedis',35, "RENNES");

CREATE TABLE APPRENANT (
id integer primary key auto_increment,
nom varchar(25) not NULL,
prenom varchar(25) not NULL,
mail varchar(30) not NULL,
pseudo_discord varchar(30) not NULL,
anciennete integer DEFAULT 0,
entreprise integer,
FOREIGN KEY (entreprise) REFERENCES ENTREPRISE (id) 
);

INSERT INTO APPRENANT (nom, prenom, mail, pseudo_discord, anciennete, entreprise) VALUES
('ADEL', 'Mehdi', 'zeadel.mehdi@gmail.com','N7Mehdi', 2, 3),
('BARRE', 'Halimata', 'halimataba54@gmail.com','halimata', NULL,4),
('BOURY', 'Damien', 'bourydamien1@gmail.com','Boury Damien', NULL,10),
('BRUN', 'Christel', 'brunday@gmail.com','Xel', 0,4),
('COURTEAU', 'Julien', 'juliencourteau22@gmail.com','juliencrt', 0,3),
('DROUILLARD', 'Erwan', 'erwan.drouillard@gmail.com ','Erwan22', 0,6),
('FIN', 'Yohann', 'yohann.fin13@gmail.com','Yohann', 4,5),
('GOYTY', 'Ellande', 'egoyty@gmail.com','ellande', 1, 1),
('GUILLAIN', 'Claire', 'claire.guillain.cg@gmail.com','Claire G', NULL,3),
('JONCOURT', 'Thomas', 'thomas.joncourt.pro1@gmail.com','Thomas J', 0,9),
('LE BRICQUER', 'Jérémy', 'jeremylebricquer@gmail.com','Jeremy', 3,3),
('LE CHAFFOTEC', 'Amaury', 'amaury.le.chaffotec2@gmail.com','AmauryLC', 0,2),
('LE COZ', 'Paul', 'plecoz.pro@gmail.com','PaulLC', 0, 8),
('MAURIAUCOURT', 'Guillaume', 'proracevdt@gmail.com','Tyrax', 0,4),
('PASQUIERS', 'Anatole', 'anatole.pasquiers1@gmail.com','Pasquiers Anatole', 0,4),
('PREVOT', 'Vincent', 'vincentprv22@gmail.com','Vincent', 0,4),
('RAOUL', 'Aurélie', 'raoulaur@gmail.com','Aurélie', 1, 7),
('SALAUN', 'Morgan', 'mowglyzer@gmail.com','Morgan S', 0,3);

SELECT * FROM ENTREPRISE;

-- Exemple de mise à jour multiple
-- UPDATE APPRENANT SET entreprise = 4 WHERE prenom = 'Christel' OR prenom = 'Guillaume' OR prenom = 'Vincent' OR prenom = 'Halimata' OR prenom = 'Anatole';

-- Vérification que tous les apprenants ont une entreprise
SELECT * 
FROM APPRENANT 
WHERE entreprise IS NULL;

-- SQL 1
SELECT APPRENANT.nom, APPRENANT.prenom, ENTREPRISE.nom, ENTREPRISE.ville
FROM APPRENANT , ENTREPRISE
WHERE APPRENANT.entreprise = ENTREPRISE.id;

-- SQL 2
SELECT APPRENANT.nom, APPRENANT.prenom, ENTREPRISE.nom, ENTREPRISE.ville
FROM APPRENANT 
INNER JOIN ENTREPRISE ON APPRENANT.entreprise = ENTREPRISE.id;

-- Nombre d'apprenants par entreprise
SELECT ENTREPRISE.id, ENTREPRISE.nom, count(APPRENANT.entreprise) AS NB_APPRENANTS 
FROM ENTREPRISE, APPRENANT
WHERE APPRENANT.entreprise = ENTREPRISE.id
GROUP BY ENTREPRISE.nom
ORDER BY ENTREPRISE.id;

-- Nombre d'apprenants par ville
SELECT E.ville, count(E.id) AS NB_ENTREPRISE FROM ENTREPRISE E
GROUP BY E.ville;

SELECT *
FROM APPRENANT A, ENTREPRISE E
WHERE E.id = A.entreprise
ORDER BY E.ville;

SELECT *
FROM APPRENANT A, 
INNER JOIN ENTREPRISE E
on A.entreprise = E.id 
ORDER BY E.ville;

SELECT nom, prenom
FROM APPRENANT a
WHERE entreprise = 3;

SELECT a.nom, prenom
FROM APPRENANT a
JOIN ENTREPRISE e on a.entreprise = e.id AND e.nom = "ORANGE";