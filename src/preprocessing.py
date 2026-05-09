import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
def load_data():
    ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', 
                          names=['userId', 'movieId', 'rating', 'timestamp'],
                          engine='python')
    movies = pd.read_csv('data/ml-1m/movies.dat', sep='::', 
                         names=['movieId', 'title', 'genres'],
                         engine='python', encoding='latin-1')
    users = pd.read_csv('data/ml-1m/users.dat', sep='::',
                        names=['userId', 'gender', 'age', 'occupation', 'zip'],
                        engine='python')
    return ratings, movies, users

# Preprocess data
def preprocess(ratings, movies):
    # Normalize ratings to 0-1
    ratings['rating'] = ratings['rating'] / 5.0

    # One-hot encode genres
    movies['genres'] = movies['genres'].str.split('|')
    genres_dummies = movies['genres'].str.join('|').str.get_dummies()
    movies = pd.concat([movies, genres_dummies], axis=1)

    # Merge ratings with movie features
    df = ratings.merge(movies[['movieId'] + list(genres_dummies.columns)], on='movieId')

    # Sort by timestamp to prevent data leakage
    df = df.sort_values('timestamp')

    # Train/val/test split (70/15/15)
    train, temp = train_test_split(df, test_size=0.3, shuffle=False)
    val, test = train_test_split(temp, test_size=0.5, shuffle=False)

    return train, val, test

if __name__ == '__main__':
    ratings, movies, users = load_data()
    train, val, test = preprocess(ratings, movies)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")