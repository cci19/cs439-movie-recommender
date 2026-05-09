import numpy as np

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0

def ndcg_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    dcg = sum([
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended_k)
        if item in set(relevant)
    ])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / idcg if idcg > 0 else 0

def evaluate(recommended, relevant, k=10):
    return {
        'precision@k': precision_at_k(recommended, relevant, k),
        'recall@k': recall_at_k(recommended, relevant, k),
        'ndcg@k': ndcg_at_k(recommended, relevant, k)
    }