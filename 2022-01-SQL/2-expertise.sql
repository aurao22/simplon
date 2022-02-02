/*
Création de la base avec les différentes tables.
*/
drop database if exists lannion;
create database lannion;

use lannion;

CREATE TABLE entreprise (
    id INT PRIMARY KEY,
    nom VARCHAR(100) NOT NULL
);

CREATE TABLE expertise (
    id INT PRIMARY KEY,
    nom VARCHAR(100) NOT NULL
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
INSERT INTO expertise VALUES
(1,'documentation'),
(2,'python');

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


# Schéma

# Comment gérer l'association multiple entre apprenant et expertise.
/*
Dans un premier temps, vous pourrez considérer que toute valeur supérieure ou égale à deux indique que l'expertise est acquise,
dans le cas contraire considérer que ce n'est pas le cas. Vous considérez ainsi les expertises de façon binaire (acquise, ou non acquise)
*/
 create table acquis(
	apprenant int,
    expertise int,
    primary key (apprenant, expertise),
    foreign key(apprenant) references apprenant(id),
    foreign key(expertise) references expertise(id)
 );
 
 INSERT INTO acquis VALUES
 (3,1),
 (4,1),
 (5,1),
 (5,2);
 
# affichage des prénoms des apprenants avec leurs expertises associées
select apprenant.prenom, expertise.nom as expertise from expertise
inner join acquis
on expertise.id = acquis.expertise
inner join apprenant
on apprenant.id = acquis.apprenant
;

