import sys
import os
sys.path.append(os.path.dirname(__file__))

from flask import Flask, jsonify, request, render_template
from recommender import HybridRecommender

app = Flask(__name__, template_folder='templates')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
engine = HybridRecommender(
    movies_path=os.path.join(BASE_DIR, 'data', 'movies.csv'),
    ratings_path=os.path.join(BASE_DIR, 'data', 'ratings.csv')
)
engine.train()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/movies')
def list_movies():
    movies = engine.movies_df[['movieId', 'title', 'genres']]
    return jsonify(movies.to_dict('records'))

@app.route('/recommend/user/<int:user_id>')
def recommend_user(user_id):
    n = request.args.get('n', 10, type=int)
    recs = engine.recommend(user_id=user_id, n=n)
    return jsonify(recs[['movieId', 'title', 'genres']].to_dict('records'))

@app.route('/recommend/movie/<int:movie_id>')
def recommend_movie(movie_id):
    n = request.args.get('n', 10, type=int)
    recs = engine.recommend(movie_id=movie_id, n=n)
    return jsonify(recs[['movieId', 'title', 'similarity_score']].to_dict('records'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
