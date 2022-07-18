
# Version python :

3.8, mais il peut arriver ponctuellement qu’on soit en 3.6 et très exceptionnellement (en cas de déterrage) en 2.7.... on fonctionne avec miniconda et on créée des environnements virtuels avec un fichier requirements.txt pour l’installation d’éventuels modules python

3.8.10

# Séparation d'un fichier texte en un fichier par phrase

csplit -f phrase -n 5  corpus_13_04_22.txt {1582}


# BONNES PRATIQUES

https://carbon.voxygen.fr/trac/voxygen/wiki/BonnesPratiquesDeDev


# SVN

## BARATINOO

WIKI : https://carbon.voxygen.fr/trac/voxygen/wiki/Compilation_Baratinoo

svn co https://carbon/svn/tts/baratinoo/trunk baratinoo

Fait une voix vide => elle ne génère pas une voix audio mais génère toutes les sorties textuelles
A lancer depuis `Baratinoo/Build` et connecter en VPN/réseau voxygen
`./makeconfig.py -d -f -u -s 24i.lin -r /mnt/nas/Synthese vide::main linx64/baratinoo.cfg`

dans Baratinoo/Build/linx64/
`./baratinoo -g 5  -l fr-FR  -v Videfr -i utf8 $filename -o bintext $newname ./baratinoo.cfg > $newnamedbg 2>&1•`  
- g => niveau de debug, 5 = max - sortie console d'erreur
- v => voice
- i => input : 

### Données :

`/mnt/nas/Linguistique/French/Corpus/corpus`

`grep --only-matching -io -E "[^a-zA-Z0-9_][a-zA-Z0-9_]{1}[\'’]" lemonde.clean.txt | sort -ui`
`grep --only-matching -E "\b[0-9]+(\.[0-9]+)*[ /][A-Za-z°]{1,3}([/\-−][A-Za-z]{1,3})?[23]?" *-01.txt | sort -ui`

### Liaisons :

https://carbon.voxygen.fr/trac/voxygen/browser/TTS/baratinoo/trunk/modules/francais/fr_liaison/fr_liaison.r

## AED : Atelier Evaluation Diagnostic

WIKI : https://carbon.voxygen.fr/trac/voxygen/wiki/Traitements%20linguistiques/AED

svn co https://carbon/svn/tts/AED/trunk AED


# NAS

/mnt/nas/Users/
/mnt/nas/Users/AurelieRaoul/
- GIT personnel

/mnt/nas/Users/Marion_O/
/mnt/nas/Users/Marion_O/Bench/Bench2022/BDS_dir/Bench_svi/

# GIT

## Création dépot

1. crééer le répertoire local
1. `cd <nom répertoire local>` 
1. `git init`
1. `git clone --bare <nom répertoire local> <nom répertoire distant (non existant)>`
1. Vérifier sur le répertoire distant que le répertoire a bien été créé
1. `git remote add <alias> <nom répertoire distant>`
1. Faire une modification en local
1. `git status`
1. `git add <nom fichier>`
1. `git commit -m 'commentaire'`
1. `git push `
1. `git push --set-upstream <alias> master` 
1. `git pull`


## Nouveau clone
1. Se positionner dans le répertoire souhaité, la commande suivante créera un nouveau dossier du nom du dépôt
1. `git clone <nom répertoire distant>`


# Commandes

#!/bin/bash -x

## Fichiers

Afficher les informations d'un fichier
`ll -h <file name>` 
Nombre de fichiers dans un répertoire
`ls -1A |wc -l`
Supprimer un dossier et ses sous dossiers
`rm -rf BDS`

## 
`chmod +x audacity-linux-3.1.3-x86_64.AppImage`
`./audacity-linux-3.1.3-x86_64.AppImage`

## Ajouter une variable d'environement :

Créer une nouvelle variable
`export GECKO_DRIVER_PATH=/home/aurelie/geckodriver`
Afficher la variable
`echo $GECKO_DRIVER_PATH`
Compléter une variable existante
`export PATH=$PATH:/home/aurelie/geckodriver`

## grep

`grep 'Liaison Rule' dbg/*`

Afficher le résultat sans préciser le fichier (-h)
`grep -h  'Liaison Rule' dbg/*`

Compter le nombre d'occurences
`grep -h  'Liaison Rule' dbg/* | sort | uniq -c`

Compter le nombre total d'occurences
`grep  'Liaison Rule' dbg/* | sort | uniq |wc -l`  

`grep -h -E "liaison [ar]" phrase00007.dbg`
`grep -h -E -io "liaison [ar]" phrase00007.dbg`
`grep --only-matching -h -E "\b[0-9]+(\.[0-9]+)*[ /][A-Za-z°]{1,3}([/\-−][A-Za-z]+ )?[23]?" */BDS/txt/*.txt`
`grep --only-matching -E "\b[0-9]+(\.[0-9]+)*[ /][A-Za-z°]{1,3}([/\-−][A-Za-z]{1,3})?[23]?" */BDS/txt/*.txt | sort -u`
`grep -h -E "(Liaison Rule .*)|(fr_liaiso.*LNGST\s+word)" phrase00332.dbg`
pour les fichiers binaires
`grep -a -h -E "(Liaison Rule .*)|(fr_liaiso.*LNGST\s+word)" phrase00332.dbg`
`grep -a -h -E "(Liaison Rule .*)|(fr_liaiso.*LNGST\s.*xsc:)" phrase03945.dbg`
`grep -a -h -E "(Liaison Rule .*)" phrase04242.dbg`

Sur les fichiers binaires :

`strings phrase03945.dbg | grep -h -E ".*Liaison Rule"`
`strings phrase03945.dbg | grep -h -E "fr_liaison.*(Liaison Rule)|(LNGST\s+word.*xsc:)"`
`strings phrase03945.dbg | grep -h -E "fr_liaison.*(Liaison Rule)|(LNGST.*word.*xsc:)"`

## iconv

`iconv -f UTF-8 -t WINDOWS-1252//TRANSLIT --output=outfile.csv inputfile.csv`
`iconv -f CP1252 -t UTF-8//TRANSLIT --output=phrase03945.utf8 phrase03945.dbg`

## Exploration

Afficher le début du fichier
`head <nom fichier>`
Afficher les 1000 premières lignes d'un fichier
`head -n 1000 <nom fichier>`
Exporter le résultat dans un fichier
`head -n 1000 <nom fichier> > <fichier_export>` 

Afficher la fin d'un fichier
`tail <nom fichier>`

Compter le nombre de ligne, mot, caractère d'un fichier
`cat log_gds.txt | sort -u | wc`
Compter le nombre de ligne, mot, caractère de plusieurs fichiers
`cat LOG_cleaned_2022-04-29-17_28_11_FULL_sentence.csv /mnt/nas/Users/Marion_O/Bench/Bench2022/BDS_dir/from_AED/*/BDS/txt/*.txt | sort -u | wc`
Réponse :
`<nb lignes>  <nb mots>  <nb caractères>`
`12561  179359 1139123`


## sed

`sed "<regex au format perl>"`
### Remplacer une chaine par une autre
`sed "s/<chaine à remplacer>/<chaine de remplacement>/g"`
Remplacer le séparateur `;` par `\t` et répéter à chaque fois `g`
regex perlienne
`sed "s/;/\t/g"`
`sed -e "s/txt/out/g"`

### Modifier dans le fichier directement
`sed -i "<regex au format perl>"`

### supprimer les 6 premières lignes d'un fichier :
`tail -n+6 monfichier`
ou mieux :
`sed '1,6d' monfichier`

## awk
Afficher les lignes
`-F ';' '(print $0)'`
Afficher la colonne 3
`-F ';' '(print $3)'`


## Copie

`cp <fichier src> <destination>`

Copie depuis ou vers un serveur
`scp <@user...> <fichier src> <destination>`

# VIM

`vim vim_temp.txt`

pour insérer des résultats de commande du terminal dans vim faire en mode normal 
`:read !macommande`
`:read !cat ../txt/phrase00392.txt`
`:read !grep -E  “(Liaison Rule .*)|(fr_liaiso.*LNGST\s+word)” phrase00332.dbg`

# Lire un fichier 

## Text
`more phrase00392.txt`

## audio
`play <nom du fichier>`
