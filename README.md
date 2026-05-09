# CS439 Movie Recommender

A hybrid Neural Collaborative Filtering (NCF) system for movie recommendations using the MovieLens 1M dataset.

## Setup
```bash
pip install pandas scikit-learn torch matplotlib
```

## Data
Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and place it in `data/ml-1m/`.

## Usage
1. Run preprocessing: `python src/preprocessing.py`
2. Open and run `notebooks/main.ipynb` top to bottom
3. Results and visualizations are saved to `results/`

## Results
| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.0003 | 0.0001 | 0.0002 |
| SVD | 0.0354 | 0.0085 | 0.0304 |
| NCF | 0.0708 | 0.0160 | 0.0641 |
