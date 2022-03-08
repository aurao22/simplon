# Git: outil de gestion de version des programmes informatiques
# Guide des commades git:
# https://rogerdudler.github.io/git-guide/index.fr.html

# Vidéo tutoriels
# (les bases) https://www.youtube.com/watch?v=gp_k0UVOYMw
# (plus complet) https://www.youtube.com/watch?v=V6Zo68uQPqE


# Création d'un dépôt "test" sur GitHub

# Copie en local d'un dépôt (repo) distant
git clone https://github.com/neodelphis/test.git

git status

ls
cd test
touch helloWorld.py
print("Hello World")

git status

git add test.py
# commit: instantané = photo du projet à un instant donné
git commit --message "Nouveau fichier test.py"


touch tbd.py
git status


nano tbd.py
# To be deleted
more tbd.py
# Liste des modifications par rapport au dernier commit
git diff


git add *
git commit -m "Ajout d un commentaire dans tbd.py"
git status


# Liste de tous les commit
git log

# Password for 'https://pjaumier85@gmail.com@github.com': 
# remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.
# Creating a personal access token
# https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

git push origin main

# Schéma dans Insights/ Network

# Le branchement - branching
git branch ma_branche
# Basculement sur la branche 'ma_branche'
git checkout ma_branche
# Ajout d'un message en fançais dans helloWorld.py

print("Bonjour tout le monde")
more helloWorld.py 

# dans la branch main
git checkout main
print("Ola mundo") # En lieu et place du hello world
git diff main..ma_branche

# Rassembler les branches
git merge ma_branche main

# Suppression des branches non utilisées
git branch -d ma_branche