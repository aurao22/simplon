CREATE DATABASE  IF NOT EXISTS `simplon_lannion` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `simplon_lannion`;
-- MySQL dump 10.13  Distrib 8.0.27, for Win64 (x86_64)
--
-- Host: localhost    Database: simplon_lannion
-- ------------------------------------------------------
-- Server version	8.0.27

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `apprenants`
--

DROP TABLE IF EXISTS `apprenants`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `apprenants` (
  `NA` int NOT NULL AUTO_INCREMENT,
  `NOM` varchar(25) NOT NULL,
  `PRENOM` varchar(25) NOT NULL,
  `MAIL` varchar(30) NOT NULL,
  `PSEUDO_DISCORD` varchar(30) NOT NULL,
  `ANCIENNETE` int DEFAULT '0',
  `NE` int DEFAULT NULL,
  PRIMARY KEY (`NA`),
  KEY `NE` (`NE`),
  CONSTRAINT `apprenants_ibfk_1` FOREIGN KEY (`NE`) REFERENCES `entreprises` (`NE`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `apprenants`
--

LOCK TABLES `apprenants` WRITE;
/*!40000 ALTER TABLE `apprenants` DISABLE KEYS */;
INSERT INTO `apprenants` 
VALUES (1,'ADEL','Mehdi','zeadel.mehdi@gmail.com','N7Mehdi',2,3),
(2,'BARRE','Halimata','halimataba54@gmail.com','halimata',NULL,4),
(3,'BOURY','Damien','bourydamien1@gmail.com','Boury Damien',NULL,10),
(4,'BRUN','Christel','brunday@gmail.com','Xel',15,4),
(5,'COURTEAU','Julien','juliencourteau22@gmail.com','juliencrt',0,3),
(6,'DROUILLARD','Erwan','erwan.drouillard@gmail.com ','Erwan22',0,6),
(7,'FIN','Yohann','yohann.fin13@gmail.com','Yohann',4,5),
(8,'GOYTY','Ellande','egoyty@gmail.com','ellande',1,1),
(9,'GUILLAIN','Claire','claire.guillain.cg@gmail.com','Claire G',NULL,3),
(10,'JONCOURT','Thomas','thomas.joncourt.pro1@gmail.com','Thomas J',0,9),
(11,'LE BRICQUER','Jérémy','jeremylebricquer@gmail.com','Jeremy',3,3),
(12,'LE CHAFFOTEC','Amaury','amaury.le.chaffotec2@gmail.com','AmauryLC',0,2),
(13,'LE COZ','Paul','plecoz.pro@gmail.com','PaulLC',0,8),
(14,'MAURIAUCOURT','Guillaume','proracevdt@gmail.com','Tyrax',0,4),
(15,'PASQUIERS','Anatole','anatole.pasquiers1@gmail.com','Pasquiers Anatole',0,4),
(16,'PREVOT','Vincent','vincentprv22@gmail.com','Vincent',0,4),
(17,'RAOUL','Aurélie','raoulaur@gmail.com','Aurélie',15,7),
(18,'SALAUN','Morgan','mowglyzer@gmail.com','Morgan S',0,3);
/*!40000 ALTER TABLE `apprenants` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `entreprises`
--

DROP TABLE IF EXISTS `entreprises`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `entreprises` (
  `NE` int NOT NULL AUTO_INCREMENT,
  `NOM` varchar(25) NOT NULL,
  `DEPARTEMENT` int DEFAULT '22',
  `VILLE` varchar(25) DEFAULT NULL,
  PRIMARY KEY (`NE`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `entreprises`
--

LOCK TABLES `entreprises` WRITE;
/*!40000 ALTER TABLE `entreprises` DISABLE KEYS */;
INSERT INTO `entreprises` VALUES (1,'Event Factory',22,'BEGARD'),(2,'Holai',22,'PLOUFRAGRAN'),(3,'Crédit Agricole',22,'PLOUFRAGRAN'),(4,'Orange',22,'LANNION'),(5,'NOKIA',22,'LANNION'),(6,'SAOTI',22,'LANNION'),(7,'VOXYGEN',22,'PLEUMEUR-BODOU'),(8,'ECO-COMPTEUR',22,'LANNION'),(9,'SF2I',22,'NOUMEA'),(10,'Labocéa',22,'PLOUFRAGAN');
/*!40000 ALTER TABLE `entreprises` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-01-10 11:17:35
