-- Création de la BD :
-- on supprime la database, on la recrée, on l'utilise
drop database if exists employesTP01;
create database employesTP01;
use employesTP01;
-- Création de la table
CREATE TABLE emp (
NE integer primary key auto_increment,
NOM varchar(10) not NULL,
FONCTION varchar(9) not NULL,
DATEMB date not NULL,
SAL float(7,2) not NULL,
COMM float(7,2),
ND integer not null,
NEchef integer
);

show tables;

desc emp;

ALTER TABLE emp MODIFY
FONCTION varchar(15) ;

DROP TABLE emp ;

DROP DATABASE if exists employesTP01;

-- création des tuples
INSERT INTO emp VALUES
(7839,'KING','PRESIDENT','1981-11-17',5000,NULL,10,NULL),
(7698,'BLAKE','MANAGER','1981-05-1',2850,NULL,30,7839),
(7782,'CLARK','MANAGER','1981-06-9',2450,NULL,10,7839),
(7566,'JONES','MANAGER','1981-04-2',2975,NULL,20,7839),
(7654,'MARTIN','SALESMAN','1981-09-28',1250,1400,30,7698),
(7499,'ALLEN','SALESMAN','1981-02-20',1600,300,30,7698),
(7844,'TURNER','SALESMAN','1981-09-8',1500,0,30,7698),
(7900,'JAMES','CLERK','1981-12-3',950,NULL,30,7698),
(7521,'WARD','SALESMAN','1981-02-22',1250,500,30,7698),
(7902,'FORD','ANALYST','1981-12-3',3000,NULL,20,7566),
(7369,'SMITH','CLERK','1980-12-17',800,NULL,20,7902),
(7788,'SCOTT','ANALYST','1982-12-09',3000,NULL,20,7566),
(NULL,'ADAMS','CLERK','1983-01-12',1100,NULL,20,7788),
(NULL,'MILLER','CLERK','1982-01-23',1300,NULL,10,7782);


UPDATE emp
SET sal = sal +100
WHERE fonction = 'CLERK' ;

DELETE FROM emp
WHERE fonction = 'CLERK' ;

SELECT * FROM emp;

SELECT * FROM emp
WHERE fonction = 'MANAGER' ;

SELECT * FROM emp
ORDER BY sal;

SELECT * FROM emp
WHERE fonction = 'MANAGER'
ORDER BY sal DESC;

SELECT avg(sal) FROM emp;

SELECT min(sal) FROM emp;

SELECT max(sal) FROM emp;