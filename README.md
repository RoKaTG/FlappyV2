# Introduction :

Le but du projet est de créer une version simple de Flappy bird. Flappy Bird consiste à éviter les obstacles en se déplaçant dans des anneaux en faisant voler ou descendre l'oiseau en tapotant sur notre écran mobile. Notre version du jeu sera une version simplifié avec un ovale représentant l'oiseau qui devra rester dans une ligne brisée sans la touchée.

Le jeu a été implémenter en Java en utilisant comme représentant du oiseau un ovale (comme dis précédemment), cette fois ci, le joueur utilisera la souris pour cliquer a chaque fois il souhaite faire voler ou descendre l’ovale.

Voici ci-dessous un exemple du début de projet **/!\ ATTENTION cela est une version non terminé du projet /!\** : ![image](https://user-images.githubusercontent.com/58750536/149665678-09252756-b25e-40aa-9455-873108c131f6.png)


# Analyse Globale :

Notre projet se compose en plusieurs fonctionnalité (trois principalement) :
Nous avons premièrement l'interface graphique, qui, dans le modèle MVC sera dans Vue, l'interface permettra le controle de la ligne brisée ainsi que l'ovale represdentant l'oiseau.
Il y ensuite la reaction automatique de la ligne brisée par rapport aux actions du joueurs (defilement etc) qui sera implémenté dans Controleur.
Puis enfin la réaction de l'ovale par rapport aux actions du joueurs qui seront aussi implémentés dans Controleur.

Dans la première partie du projet (séance 1), nous nous intéressons uniquement à un sous-ensemble de fonctionnalités :

° Création d’une fenêtre dans laquelle est dessiné l’ovale.

° Déplacement de l’ovale vers le haut lorsqu’on clique dans l’interface.

Ces deux sous-fonctionnalités sont prioritaires et simples à réaliser car elles utilisent l'Api Swing de base necessitant seulement la bibliothèque java.

Dans la deuxième partie du projet (séance 2), nous nous intéressons uniquement aux fonctionalités suivantes :

° Le déplacement de l'ovale et la réaction de la fenêtre par rapport à celle ci.

° La génération infinie d'une ligne brisée.


# Plan de développement :

## Interface graphique :

Le temps de travail estimé : 3h

— Analyse du problème : 30 min

— Acquisition de compétences en Swing : 60 min

— Conception et test de la fenêtre : 30 min

— Conception de l'ovale : 30 min

— Documentation du projet : 30 min

## Déplacement de l’ovale :

Le temps de travail estimé : 1h20

— Analyse du problème : 20 min

— Conception des mouvements et des réactions : 1h

## La ligne brisée :

Le temps de trvail estimé : 1h15

— Analyse du problème : 15 min

— Conception, teste : 1h


# Conception générale :
Nous avons adopté le motif MVC pour le développement de notre interface graphique pour plusieurs raisons logique : 

Le MVC permet un travail propre et efficace sur les interfaces graphiques avec notamment la modification de l'environement produit après execution du programme (dans les jeux notamments) donc l'interface graphice produit par Affichage doit pouvoir subir des adaptations par rapport mouvement fait par l'ovale et la ligne brisée d'où l'utilisation d'un Controleur, Vue mais aussi Modele car nous allons à la fin du projet implémenter une interface graphique en 2D.


# Conception détaillée :
En cours de production ...


# Documentation utilisateur :
° Prérequis : Java avec un IDE (ou Java tout seul si vous avez fait un export en .jar exécutable)

° Mode d’emploi (cas IDE) : Importez le projet dans votre IDE, sélectionnez la classe Main à la racine du projet puis « Run as Java Application ». Cliquez sur la fenêtre pour faire monter l’ovale.

° Mode d’emploi (cas .jar exécutable) : double-cliquez sur l’icône du fichier .jar. Cliquez sur la fenêtre pour faire monter l’ovale.


# Documentation developpeur :
En cours de production...


# Conclusion et perspectives :
Soon...
