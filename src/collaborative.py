import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilter:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.trained = False

    def train(self, ratings_df):
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.trained = True
        print("✅ Collaborative model trained!")

    def recommend_for_user(self, user_id, movies_df, n=10):
        if user_id not in self.user_item_matrix.index:
            return movies_df.head(n)
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        sim_scores = self.user_similarity[user_idx]
        weighted = np.dot(sim_scores, self.user_item_matrix.values)
        movie_scores = pd.Series(weighted, index=self.user_item_matrix.columns)
        top_ids = movie_scores.nlargest(n).index.tolist()
        return movies_df[movies_df['movieId'].isin(top_ids)]
