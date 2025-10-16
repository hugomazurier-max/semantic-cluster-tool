from typing import Dict, Any, List
import pandas as pd
import numpy as np
from collections import Counter
import re

def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-zàâäéèêëïîìôöùûüç\-']{2,}", text.lower())
    return toks

def term_targets_from_tfidf(tfidf_vec, tfidf_X_row, vocab, per_page_terms=40, target_len_words=1200):
    vec = tfidf_X_row.toarray().ravel()
    idx = vec.argsort()[::-1][:per_page_terms]
    terms = [(vocab[i], float(vec[i])) for i in idx if vec[i] > 0]
    if not terms:
        return []
    weights = np.array([w for _, w in terms])
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    weights = 0.6*weights + 0.4
    per_1000 = (weights * 3.0)
    targets = []
    for (t, w), occ in zip(terms, per_1000):
        mn = max(1, int(round(occ)))
        mx = mn + (1 if occ > 2.0 else 0)
        targets.append({"terme": t, "poids": round(w,3), "cible_min_1000": mn, "cible_max_1000": mx})
    return targets

def coverage_score(page_text: str, terms: List[str]) -> float:
    toks = tokenize(page_text)
    bag = set(toks)
    hits = sum(1 for t in terms if t in bag)
    return round(100.0 * hits / max(1, len(terms)), 1)

def generate_briefs_pro(analysis: Dict[str, Any], clusters_df, target_len_words: int = 1200, per_page_terms: int = 40) -> pd.DataFrame:
    pages = analysis["pages"]
    tfidf_vec = analysis["tfidf_vec"]
    X = analysis["tfidf_X"]
    vocab = analysis["vocab"]
    rows = []
    for i, p in enumerate(pages):
        targets = term_targets_from_tfidf(tfidf_vec, X[i], vocab, per_page_terms, target_len_words)
        terms = [t["terme"] for t in targets]
        score = coverage_score(p["text"], terms)
        for rank, t in enumerate(targets, start=1):
            rows.append({
                "URL": p["url"],
                "Titre": p.get("title",""),
                "Cluster": int(clusters_df.loc[clusters_df["url"]==p["url"], "cluster"].values[0]) if not clusters_df.empty else -1,
                "Priorité": 1 if rank <= int(per_page_terms*0.4) else (2 if rank <= int(per_page_terms*0.8) else 3),
                "Terme": t["terme"],
                "Cible (min/1000 mots)": t["cible_min_1000"],
                "Cible (max/1000 mots)": t["cible_max_1000"],
                "Section suggérée": "Intro/H2/H3/FAQ",
                "Note": ""
            })
        rows.append({
            "URL": p["url"], "Titre": p.get("title",""), "Cluster": int(clusters_df.loc[clusters_df["url"]==p["url"], "cluster"].values[0]) if not clusters_df.empty else -1,
            "Priorité": "", "Terme": "__SCORE_COUVERTURE__", "Cible (min/1000 mots)": score, "Cible (max/1000 mots)": "", "Section suggérée": "", "Note": "Score de couverture (%) des termes proposés"
        })
    return pd.DataFrame(rows)

def export_briefs_pro_csv(df: pd.DataFrame, path="exports/briefs_pro.csv"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
