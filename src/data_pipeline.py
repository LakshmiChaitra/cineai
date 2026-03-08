import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPipeline:
    def __init__(self, movies_path, ratings_path):
        self.movies_path = movies_path
        self.ratings_path = ratings_path

    def load_movies(self):
        movies = pd.read_csv(self.movies_path)
        movies['genres_list'] = movies['genres'].str.split('|')
        return movies

    def load_ratings(self):
        ratings = pd.read_csv(self.ratings_path)
        scaler = MinMaxScaler()
        ratings['rating_normalized'] = scaler.fit_transform(
            ratings[['rating']]
        )
        return ratings

    def build_user_item_matrix(self):
        ratings = self.load_ratings()
        matrix = ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        return matrix