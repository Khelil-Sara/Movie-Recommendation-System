import pandas as pd
import json
import os

# 1 Data Integration 
# Fusionner les données de Haythem avec les crédits
credits = pd.read_csv(os.path.join(path, 'tmdb_5000_credits.csv'))
md = md.merge(credits, on='title')

# 2Cleaning Functions

# Fonction pour extraire le réalisateur
def get_director(obj):
    L = []
    try:
        data = json.loads(obj)
        for i in data:
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

# Fonction pour supprimer les espaces (Crucial pour le modèle)
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

# 3Processing Features

# Extraction des acteurs (Top 3)
md['cast'] = md['cast'].apply(lambda x: [i['name'] for i in json.loads(x)][0:3] if isinstance(x, str) else [])

# Extraction du réalisateur
md['crew'] = md['crew'].apply(get_director)

# Nettoyage des mots-clés (Keywords)
md['keywords'] = md['keywords'].apply(lambda x: [i['name'] for i in json.loads(x)] if isinstance(x, str) else [])

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
new_df = md[['id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print("Preprocessing complete. File 'new_df' is ready for the Model member.")
