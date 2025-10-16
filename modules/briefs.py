import pandas as pd
import numpy as np
from collections import Counter
import os

def generate_briefs(analysis, clusters_df, cfg):
    pages = analysis["pages"]
    vocab = analysis["vocab"]
    X = analysis["tfidf_X"]
    ents = analysis["ents"]
    top_ngrams = analysis["top_ngrams"]

    per_page_terms = cfg["briefs"]["per_page_terms"]
    ess = int(per_page_terms*cfg["briefs"]["essential_ratio"])
    sec = int(per_page_terms*cfg["briefs"]["secondary_ratio"])
    opp = per_page_terms - ess - sec

    rows = []
    for i,p in enumerate(pages):
        # Top TF-IDF features pour la page i
        vec = X[i].toarray().ravel()
        top_idx = vec.argsort()[::-1][:per_page_terms*2]
        tfidf_terms = [vocab[j] for j in top_idx]
        # N-grams & entités
        ng = [t for t,_ in top_ngrams[i]]
        en = [e for e,_ in ents[i]]

        # Fusion simple avec déduplication
        fused = []
        seen = set()
        for term in tfidf_terms + ng + en:
            t = term.strip().lower()
            if t and t not in seen:
                seen.add(t)
                fused.append(t)
            if len(fused)>=per_page_terms:
                break

        # Priorités
        essentials = fused[:ess]
        secondaries = fused[ess:ess+sec]
        opportunities = fused[ess+sec:ess+sec+opp]

        def add_rows(terms, prio):
            for t in terms:
                rows.append({
                    "URL": p["url"],
                    "Titre": p.get("title",""),
                    "Terme": t,
                    "Priorité": prio,
                    "Type": "mot-clé/entité/ngram",
                    "Section suggérée": "H2/H3",
                    "Ancre interne candidate": "",
                    "Note d’intégration": ""
                })

        add_rows(essentials, 1)
        add_rows(secondaries, 2)
        add_rows(opportunities, 3)

    return pd.DataFrame(rows)

def export_briefs_csv(df: pd.DataFrame, path="exports/briefs_lexicaux.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
