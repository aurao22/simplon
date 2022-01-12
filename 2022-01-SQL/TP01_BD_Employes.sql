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
