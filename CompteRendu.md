# Compte rendu TP NLP – Analyse des embeddings et classification émotionnelle
** par Guilhem DUPUY, Anh Duy VU, Artus BLETON **

## 1. Fonctionnalités développées

#### Organisation de notre code : 
- `RNN_Model.py` : définition de notre modèle neuronal dans la classe `CustomRNN_Manual`
- `p1.py` : regroupe toutes les atres fonctions et objets personnalisés que nous utilisons dans le projet
- `main.py`: contient notre travail pour la création et l'entraînement du modèle de classification des émotions
- `main_embeddings_visu.py`: travail sur la partie visualisation de la représentation des mots

#### Ce que nous avons mis en place : 
- **Chargement et prétraitement des données** :  
  - Lecture des fichiers train/test (`load_file`)  
  - Tokenisation simple (`tokenizer`)  
  - Undersampling aléatoire pour équilibrer les classes (`undersample_dataset_random`)  
- **Construction des datasets PyTorch** :  
  - classe `EmotionDataset` pour classification supervisée  
- **RNN personnalisé** :  
  - `CustomRNN_manual` avec embedding, couche linéaire, normalisation, dropout et connexion résiduelle
  - Initialisation Xavier, clipping des gradients, support mini-batch  
- **Apprentissage supervisé et auto-supervisé** :  
  - Boucles d’entraînement optimisées avec Adam
  - Support pour validation, calcul de la précision et affichage des courbes d’apprentissage
- **Visualisation des embeddings** :  
  - PCA et t-SNE
  - Visualisation avec Plotly, affichage du mot au survol

## 2 - Notre meilleur modèle : 

#### Meilleur hyperparamétrage
- **Embedding size** : 128  
- **Hidden size** : 128  
- **Sequence length max** : 20  
- **Batch size** : 10
- **Learning rate** : 0.001  
- **Nb epochs** : 50

**performance obtenue : ** 

INSERER LES VISUALISATIONS ICI


## 3 - Analyse des embeddings : 

#### Fonctionnalités développées / testées : 
- Travail rassemblé dans le fichier "main_embeddings_visu.py"
- Réduction de dimension (méthodes PCA et t-SNE testées)
- Visualisation des résultats via l'utilistion de Plotly

#### Choix : 
- la réduction de dimensions PCA : 1 dimension semblait prédominer sur la PCA, ce qui fait que les résultats étaient plus alignés donc moins lisibles
- La réduction par t-SNE donnait des résultats plus analysables, et répartis dans l'espace. Idéal pour la recherche d'illustrations de la théorie des analogies vectorielles dans les embeddings.
- Les premières analyses ont été effectuées sur les 200 mots les plus fréquents. Ces analyses étaient polluées par la présence de mots trop communs et de prépositions, (tels que I, You, if, etc ...). Nous avons donc exclu les 50 mots les plus fréquents, pour nous concentrer sur les 200 d'après, plus porteur de sens.
- En augmentant à 150 le nombre de mots exclus, certains clusters se définissent encore plus clairement. Ces 2 représentations sont enregistrées dans les fichiers "EmbeddingSpace_50wordsExcluded.png" et "EmbeddingSpace_150wordsExcluded.png"

#### Sur la visualisation, identification des principaux clusters : 
- [-25; 65] : cluster interprété comme celui adjectifs positifs, attribuables à un partenaire romantique (sweet, caring, loving, supporting, mais aussi hot et horny apparemment). Tous ces adjectifs sont extrèmement regroupés dans l'espace d'embedding réduit, et identifiés comme quasiment identiques par notre modèle RNN. C'est le cluster le mieux défini visuellement.
- [-58; -9] : cluster des adjectifs associés au danger (cold, agitated, angry, dangerous, irritated)
- [-10; 2] : cluster de la timidité (Insecure, unsure, intimidated, anxious ...)
- Note : les clusters, même plus petits, semblent finalement rendre compte de la classe grammaticale des mots : adjectifs, verbes, noms ...

#### Illustration trouvées du théorème des analogies vectorielles dans les embeddings :
- Le vecteur du mot "No" semble utiliser pour qualifier plusieurs types de relations dans notre espace (vecteur [37;-37])
- Exemple 1 : utilisation de négation classique. Une distance similaire au vecteur "no" entre "didn't" et "doing".
- Exemple 2 : utilisation pour distinguer un mot et son contraire : relation observée entre "good" et "bad", entre "Always" et "Never".

#### Conclusions : 
- Globalement, ces visualisations donnent une bonne intuition de la façon dont le modèle encode le sens des mots et des relations entre eux. On retrouve bien certaines relations classiques comme la négation, ce qui confirme que le modèle capture des analogies entre mots.
- Cet exemple illustre l'intéret du part-of-speech tagging (POS-tagging) dans le NLP : notre modèle semble accorder beaucoup d'importance à la classe grammaticale des mots. Avoir des données déjà étiquettées doit permettre d'améliorer et accélérer sensiblement l'apprentissage.
- La réduction par t-SNE semble plus adaptée à cet usage. D’après nos recherches, cela vient du fait que cette méthode est non-linéaire et cherche à préserver les distances locales tout en étirant les zones de faible densité. t-SNE est donc plus susceptible de révéler des nuances subtiles, que PCA pourrait “écraser” par son approche linéaire brute.
- L’apprentissage auto-supervisé (contexte => mot) devrait permettre de renforcer ces liens et de rendre les embeddings encore plus représentatifs.
