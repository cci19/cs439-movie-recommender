import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

# ── Baseline 1: Popularity-based ──────────────────────────────────────────────
class PopularityRecommender:
    def fit(self, train):
        self.popular_movies = (
            train.groupby('movieId')['rating']
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def recommend(self, n=10):
        return self.popular_movies[:n]


# ── Baseline 2: SVD Collaborative Filtering ───────────────────────────────────
class SVDRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)
        self.user_enc = LabelEncoder()
        self.movie_enc = LabelEncoder()

    def fit(self, train):
        train = train.copy()
        train['user_idx'] = self.user_enc.fit_transform(train['userId'])
        train['movie_idx'] = self.movie_enc.fit_transform(train['movieId'])

        n_users = train['user_idx'].max() + 1
        n_movies = train['movie_idx'].max() + 1

        # Build user-item matrix
        self.matrix = np.zeros((n_users, n_movies))
        for _, row in train.iterrows():
            self.matrix[int(row['user_idx']), int(row['movie_idx'])] = row['rating']

        self.user_factors = self.svd.fit_transform(self.matrix)
        self.movie_factors = self.svd.components_.T

    def predict(self, user_idx, movie_idx):
        return np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])


# ── Main Model: Neural Collaborative Filtering (NCF) ─────────────────────────
class NCF(nn.Module):
    def __init__(self, n_users, n_movies, n_genres, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2 + n_genres, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_idx, movie_idx, genre_features):
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        x = torch.cat([user_emb, movie_emb, genre_features], dim=1)
        return self.fc_layers(x).squeeze()