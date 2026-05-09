import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    ratings = pd.read_csv(os.path.join(BASE_DIR, 'data/ml-1m/ratings.dat'), sep='::', 
                          names=['userId', 'movieId', 'rating', 'timestamp'],
                          engine='python')
    movies = pd.read_csv(os.path.join(BASE_DIR, 'data/ml-1m/movies.dat'), sep='::', 
                         names=['movieId', 'title', 'genres'],
                         engine='python', encoding='latin-1')
    users = pd.read_csv(os.path.join(BASE_DIR, 'data/ml-1m/users.dat'), sep='::',
                        names=['userId', 'gender', 'age', 'occupation', 'zip'],
                        engine='python')
    return ratings, movies, users

def preprocess(ratings, movies):
    ratings['rating'] = ratings['rating'] / 5.0
    movies['genres'] = movies['genres'].str.split('|')
    genres_dummies = movies['genres'].str.join('|').str.get_dummies()
    movies = pd.concat([movies, genres_dummies], axis=1)
    df = ratings.merge(movies[['movieId'] + list(genres_dummies.columns)], on='movieId')
    df = df.sort_values('timestamp')
    train, temp = train_test_split(df, test_size=0.3, shuffle=False)
    val, test = train_test_split(temp, test_size=0.5, shuffle=False)
    return train, val, test

if __name__ == '__main__':
    ratings, movies, users = load_data()
    train, val, test = preprocess(ratings, movies)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")