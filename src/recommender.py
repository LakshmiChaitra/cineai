import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from collaborative import CollaborativeFilter
from content_based import ContentBasedFilter
from data_pipeline import DataPipeline

class HybridRecommender:
    def __init__(self, movies_path, ratings_path):
        self.pipeline = DataPipeline(movies_path, ratings_path)
        self.collab = CollaborativeFilter()
        self.content = ContentBasedFilter()
        self.movies_df = None

    def train(self):
        self.movies_df = self.pipeline.load_movies()
        ratings_df = self.pipeline.load_ratings()
        self.collab.train(ratings_df)
        self.content.fit(self.movies_df)
        print("✅ Hybrid recommender ready!")

    def recommend(self, user_id=None, movie_id=None, n=10):
        if user_id:
            return self.collab.recommend_for_user(
                user_id, self.movies_df, n
            )
        elif movie_id:
            return self.content.get_similar_movies(movie_id, n)
        else:
            raise ValueError("Provide user_id or movie_id")