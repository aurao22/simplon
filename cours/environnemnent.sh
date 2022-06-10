# Gestion des environnements python

# Une solution intégrée à python: virtualenv
# https://docs.python.org/3/tutorial/venv.html

# La solution conseillée: utilisation de conda
# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


# Création d'environnement
conda create --name test

conda create -p "C:\Users\User\WORK\workspace-ia\PROJETS\ema_lannuontimes" --name ema_lannuontimes
conda env list

# Se placer dans l'environnement
conda activate test
# (test) ➜  ~

# Installation de librairies, par exemple beautifulsoup
conda install -c anaconda beautifulsoup4

# Sauvegarde des caractérisques d'un environnment pour le reproduire
conda env export > test.yml

# Sortie de l'environnement
conda deactivate

# Suppression de l'environnement
conda env remove --name test

# Création d'environnement à partir d'un fichier yaml
conda env create -f test.yml


# Ajout de l'environnement à Jupyter
# Il faut que Jupyter soit installé
# cf "method 2" dans
# https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874
(new-env)$ conda install ipykernel
(new-env)$ipython kernel install --user --name=test

# Créer un environnement virtuel
python -m venv <myenvname>

# Please note that venv does not permit creating virtual environments with other versions of Python. For that, install and use the virtualenv
virtualenv --python=/usr/bin/python2.6 <path/to/new/virtualenv/>