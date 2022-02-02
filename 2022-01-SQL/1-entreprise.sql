/*
Création de la base avec les différentes tables.
*/
drop database if exists lannion;
create database lannion;

use lannion;

CREATE TABLE entreprise (
    id INT PRIMARY KEY,
    nom VARCHAR(30) NOT NULL
);

CREATE TABLE apprenant (
    id INT PRIMARY KEY,
    prenom VARCHAR(50) NOT NULL,
    nom VARCHAR(100) NOT NULL,
    mail VARCHAR(100),
    discord VARCHAR(50),
    anciennete INT,
    entreprise INT,
    FOREIGN KEY (entreprise) REFERENCES entreprise (id)
);

/*
Alimentation des tables
*/
INSERT INTO entreprise VALUES
(1,'Event Factory'),
(2,'Hoali'),
(3,'Crédit Agricole');


INSERT INTO apprenant VALUES
(1,'Ellande','Goyty','egoyty@gmail.com', 'ellande', 1, 1),
(2,'Amaury','Le Chaffotec','amaury.le.chaffotec2@gmail.com', 'AmauryLC', null, 2),
(3,'Claire','Guillain','claire.guillain.cg@gmail.com', 'Claire G', null, null),
(4,'Julien','Courteau', 'juliencourteau22@gmail.com', 'juliencrt', 0, 3),
(5,'Yohann','Fin', 'yohann.fin13@gmail.com', 'Yohann', 4, 3);

# Nom et prénom des apprenants avec leurs entreprises associées.
select apprenant.nom, apprenant.prenom, entreprise.nom
from apprenant
inner join entreprise
on apprenant.entreprise = entreprise.id;
/* !!!
On voit que cela fait disparaître les apprenants qui n'ont pas d'entreprise associée. Il faudra voir l'usage des jointures externes pour résoudre ce problème.
*/


# Les noms et prénoms des personnes allant effectuer leur stage chez Crédit Agricole
select apprenant.nom, apprenant.prenom, entreprise.nom
from apprenant
inner join entreprise
on apprenant.entreprise = entreprise.id
where entreprise.nom = 'Crédit Agricole';


# Modifier la table pour pouvoir intégrer le secteur d'activité de l’entreprise
alter table entreprise
add secteur varchar(255);

desc entreprise;