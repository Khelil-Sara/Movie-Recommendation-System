import pandas as pd
import json
import os
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DATA
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
md = pd.read_csv(os.path.join(path, 'tmdb_5000_movies.csv'))
credits = pd.read_csv(os.path.join(path, 'tmdb_5000_credits.csv'))
md = md.merge(credits, on='title')

def get_director(obj):
    try:
        data = json.loads(obj)
        for i in data:
            if i['job'] == 'Director':
                return [i['name']]
    except:
        pass
    return []

def collapse(L):
    return [i.replace(" ", "") for i in L]

md['cast']     = md['cast'].apply(lambda x: [i['name'] for i in json.loads(x)][0:3] if isinstance(x, str) else [])
md['crew']     = md['crew'].apply(get_director)
md['keywords'] = md['keywords'].apply(lambda x: [i['name'] for i in json.loads(x)] if isinstance(x, str) else [])
md['genres']   = md['genres'].apply(lambda x: [i['name'] for i in json.loads(x)] if isinstance(x, str) else [])
md['genres']   = md['genres'].apply(collapse)
md['keywords'] = md['keywords'].apply(collapse)
md['cast']     = md['cast'].apply(collapse)
md['crew']     = md['crew'].apply(collapse)
md['overview'] = md['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
md['tags']     = md['overview'] + md['genres'] + md['keywords'] + md['cast'] + md['crew']

new_df = md[['id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
print("✅ Preprocessing done:", new_df.shape)

# MODEL
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()

def get_recommendations(title, n=10):
    if title not in indices:
        return f"❌ Film '{title}' non trouvé."
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 3) for i in sim_scores]
    return pd.DataFrame({'title': new_df['title'].iloc[movie_indices].values, 'similarity_score': scores})

# TEST
print(get_recommendations("The Dark Knight"))