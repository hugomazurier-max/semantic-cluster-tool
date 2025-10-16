import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def cluster_pages(analysis, cfg):
    texts = [p["text"] for p in analysis["pages"]]
    model = SentenceTransformer(cfg["similarity"]["model_name"])
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    sim = cosine_similarity(emb)

    # Simple KMeans auto (k = sqrt(N) approx, min 2)
    n = max(2, int(len(texts)**0.5))
    km = KMeans(n_clusters=n, n_init="auto", random_state=42)
    labels = km.fit_predict(emb)

    df = pd.DataFrame({
        "url": [p["url"] for p in analysis["pages"]],
        "title": [p["title"] for p in analysis["pages"]],
        "cluster": labels
    })
    return df, sim
