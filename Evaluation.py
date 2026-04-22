

indices = pd.Series(md.index, index=md['title']).drop_duplicates()

def evaluate_model(test_movies):
    success = 0

    for movie in test_movies:
        if movie in indices:
            recs = get_recommendations(movie)

            print(f"\n {movie} -> Recommendations:")
            print(recs['title'].tolist()[:5]) 

            success += 1
        else:
            print(f" Movie not found: {movie}")

    print("\n Evaluation Summary:")
    print(f"Found: {success} / {len(test_movies)}")