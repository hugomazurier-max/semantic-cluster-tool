import re
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def normalize_kw(k: str) -> str:
    k = k.strip().lower()
    k = re.sub(r"\s+", " ", k)
    return k

def cluster_keywords(keywords: List[str], n_clusters: int = None, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> pd.DataFrame:
    kws = [normalize_kw(k) for k in keywords if k and k.strip()]
    kws = list(dict.fromkeys(kws))
    if len(kws) < 2:
        return pd.DataFrame(columns=["keyword", "cluster"])
    model = SentenceTransformer(model_name)
    emb = model.encode(kws, normalize_embeddings=True)
    if n_clusters is None:
        n_clusters = max(2, int(len(kws) ** 0.5))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(emb)
    df = pd.DataFrame({"keyword": kws, "cluster": labels})
    # pillar + satellites
    outlines = []
    for c, sub in df.groupby("cluster"):
        pillar = max(sub["keyword"], key=lambda s: len(s.split()))
        satellites = [k for k in sub["keyword"].tolist() if k != pillar][:12]
        h2s = [f"Guide: {pillar.title()}", "Matériel et supports", "Techniques et erreurs fréquentes", "FAQ et cas particuliers"]
        outlines.append({"cluster": int(c), "pillar": pillar, "satellites": satellites, "h2": h2s})
    out_df = pd.DataFrame(outlines)
    return df.merge(out_df, on="cluster", how="left")

def export_cocons_to_csv(df: pd.DataFrame, path: str = "exports/cocons_keywords.csv"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
