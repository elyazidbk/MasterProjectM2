Au début de chaque fichier jupyter, ajouter dans un bloc:
import os, sys
target = "BacktesterOHLCV"
while os.path.basename(os.getcwd()) != target:
    os.chdir("..")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

ça permet d'initialiser le bon dossier pour les appels relatifs de package.

Aussi, mettre le dossier stock_market_data entier dans le dossier data. le fichier .gitignore permet d'éviter de push ce dossier vu la taille, mais ça permet de le garder en local.