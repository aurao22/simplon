Installation WSL (Ubuntu) sous Windows : (https://docs.microsoft.com/en-us/windows/wsl/install)
- wsl --install > dans une invite de commande administrateur
- redémarrer le PC > installation de mise à jour
- ouvrir console Ubuntu > fin de l'installation, attention au login et mdp que vous choisissez
- pwd pour vérifier que ça fonctionne > doit vous afficher le chemin courant
- python3 -- version pour vérifier la version de python
- pip freeze pour voir ce qui est installé (si erreur, cf. installation de pip ci-dessous)

Installation pip sous ubuntu
sudo apt update && upgrade
sudo apt install python3 python3-pip ipython3

Export de la liste des modules pythons installés (dans votre environnement python)

# Export des modules installés
pip freeze > requirements.txt

Pour accéder à vos documents sous Ubuntu WSL, copier / coller votre fichier requirements.txt dans le répertoire :
\\wsl.localhost\Ubuntu\home\<nom utilisateur>

# Installation des modules à partir du fichier
python3 -m pip install -r requirements.txt

Dans VS Code :
- installer le plugin : Remote - WSL
- ouvrir (en bas à gauche) une nouvelle fenêtre WSL
- installer les plugins souhaités (modifié)
