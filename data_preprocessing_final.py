import pandas as pd
import json
import os
import ast

path = "."
md = pd.read_csv(os.path.join(path, 'tmdb_5000_movies.csv'))

# 1 Data Integration 
# Fusionner les données de Haythem avec les crédits
credits = pd.read_csv(os.path.join(path, 'tmdb_5000_credits.csv'))
md = md.merge(credits, on='title')

# 2Cleaning Functions

# Fonction pour extraire le réalisateur
def get_director(obj):
    L = []
    try:
        data = ast.literal_eval(obj) if isinstance(obj, str) else []
        for i in data:
            if i.get('job') == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

# Fonction pour supprimer les espaces (Crucial pour le modèle)
def collapse(L):
    if isinstance(L, list):
        return [i.replace(" ","") for i in L]
    return []

# Extraction des acteurs (Top 3)
def convert_cast(obj):
    try:
        data = ast.literal_eval(obj)
        return [i['name'] for i in data[:3]]
    except:
        return []

# Nettoyage des mots-clés (Keywords)
def convert_generic(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

# 3Processing Features

md['cast'] = md['cast'].apply(convert_cast)

# Extraction du réalisateur
md['crew'] = md['crew'].apply(get_director)

md['genres'] = md['genres'].apply(convert_generic)
md['keywords'] = md['keywords'].apply(convert_generic)

# Suppression des espaces pour éviter la confusion du modèle
# Exemple: "Science Fiction" -> "ScienceFiction"
md['genres'] = md['genres'].apply(collapse)
md['keywords'] = md['keywords'].apply(collapse)
md['cast'] = md['cast'].apply(collapse)
md['crew'] = md['crew'].apply(collapse)

#4Tags Creation 

# Préparation de la description (Overview)
md['overview'] = md['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Création de la colonne magique 'tags'
md['tags'] = md['overview'] + md['genres'] + md['keywords'] + md['cast'] + md['crew']

# Création du DataFrame final pour le Membre 3
new_df = md[['id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print("Preprocessing complete. File 'new_df' is ready for the Model member.")
