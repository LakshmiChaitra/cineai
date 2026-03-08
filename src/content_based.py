import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFilter:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        self.movies_df = None

    def fit(self, movies_df):
        self.movies_df = movies_df.copy()
        self.movies_df['features'] = (
            self.movies_df['genres'].fillna('') + ' ' +
            self.movies_df['title'].fillna('')
        )
        tfidf_matrix = self.tfidf.fit_transform(
            self.movies_df['features']
        )
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        print(f"✅ Content model built!")

    def get_similar_movies(self, movie_id, n=10):
        idx = self.movies_df[
            self.movies_df['movieId'] == movie_id
        ].index[0]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:n+1]
        movie_indices = [s[0] for s in scores]
        similar = self.movies_df.iloc[movie_indices].copy()
        similar['similarity_score'] = [s[1] for s in scores]
        return similar