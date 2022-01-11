-- Création de la BD :
-- on supprime la database, on la recrée, on l'utilise
drop database if exists employesTP01;
create database employesTP01;
use employesTP01;

CREATE USER bertrand@localhost
IDENTIFIED BY 'abcde';

mysql -ubertrand -pabcde

ALTER USER 'bertrand'@'localhost'
IDENTIFIED BY '';

DROP USER 'bertrand'@'localhost';

GRANT ALL PRIVILEGES
ON employesTP01.*
TO bertrand@localhost;

REVOKE ALL PRIVILEGES
ON *.*
FROM bertrand@localhost

SHOW GRANTS;

SHOW GRANTS
FOR bertrand@localhost;