from typing import Dict, Any, List
import pandas as pd
import re
from collections import Counter

SECTION_MAP = [
    ("Intro", ["définition","introduction","présentation","pourquoi"]),
    ("H2: Matériel/Supports", ["matériel","support","toile","papier","bois","plexiglas","pinceau","médiums","vernis"]),
    ("H2: Méthode/Techniques", ["technique","méthode","pas à pas","relief","texture","pouring","geste"]),
    ("H2: Erreurs/Conseils", ["erreur","astuce","conseil","éviter","piège","compatibilité"]),
    ("H2: FAQ", ["combien","comment","peut on","peut-on","c'est quoi","qu'est ce que","qu’est-ce que"]),
    ("Conclusion", ["récapitulatif","à retenir","conclusion"]),
    ("Légende d'image", ["légende","photo","détail"]),
]

def noun_phrases_like(text: str, max_len: int = 6) -> List[str]:
    tokens = re.findall(r"[a-zàâäéèêëïîìôöùûüç0-9\-']{2,}", text.lower())
    cands = []
    for n in range(1, min(max_len, 5)+1):
        for i in range(len(tokens)-n+1):
            chunk = " ".join(tokens[i:i+n])
            if len(chunk) >= 6 and not chunk.isdigit():
                cands.append(chunk)
    top = [t for t,_ in Counter(cands).most_common(100)]
    blacklist = set(["et","ou","les","des","de","la","le","un","une","du","au","aux","pour","avec","sur","dans","à"])
    top = [t for t in top if t.split()[0] not in blacklist]
    return top[:50]

def guess_section(term: str) -> str:
    t = term.lower()
    for sec, cues in SECTION_MAP:
        if any(cue in t for t in [term.lower()] for cue in cues):
            return sec
    if re.match(r"^(comment|combien|peut[- ]on|peut-on|c[’']?est quoi|qu[’']?est-ce que)", t):
        return "H2: FAQ"
    return "H2: Méthode/Techniques"

def enrich_page(analysis: Dict[str, Any], clusters_df, links_df: pd.DataFrame, per_page_terms: int = 60) -> pd.DataFrame:
    pages = analysis["pages"]
    vocab = analysis["vocab"]
    X = analysis["tfidf_X"]
    ents = analysis["ents"]
    ngrams = analysis["top_ngrams"]

    rows = []
    for i, p in enumerate(pages):
        url = p["url"]
        title = p.get("title","")
        cluster = int(clusters_df.loc[clusters_df["url"]==url, "cluster"].values[0]) if not clusters_df.empty else -1

        vec = X[i].toarray().ravel()
        idx = vec.argsort()[::-1][:per_page_terms]
        tfidf_terms = [vocab[j] for j in idx if vec[j] > 0]

        cooc = [ng for ng,_ in ngrams[i]][:30]

        ents_i = [e for e,_ in ents[i]]
        ents_i = list(dict.fromkeys(ents_i))[:20]

        qs = re.findall(r"(?:Comment|Combien|Peut[- ]on|Peut-on|C[’']?est quoi|Qu[’']?est-ce que)[^?]+\?", p["text"], flags=re.I)
        qs = [re.sub(r"\s+", " ", q.strip()) for q in qs][:10]

        nps = noun_phrases_like(p["text"])[:30]

        anchors = []
        if links_df is not None and not links_df.empty:
            subset = links_df[links_df["source_url"]==url].head(8)
            for _, r in subset.iterrows():
                tgt_title = r.get("target_title", "") if "target_title" in subset.columns else ""
                if not tgt_title and "Cible titre" in subset.columns:
                    tgt_title = r["Cible titre"]
                if not tgt_title and "target_url" in subset.columns:
                    try:
                        slug = r["target_url"].strip("/").split("/")[-1]
                        tgt_title = slug.replace("-", " ").title()
                    except Exception:
                        pass
                if tgt_title:
                    anchors.append(tgt_title)

        buckets = [
            ("mot-clé", tfidf_terms[:per_page_terms]),
            ("cooccurrence", cooc),
            ("entité", ents_i),
            ("question", qs),
            ("syntagme", nps),
        ]

        rank = 1
        for typ, terms in buckets:
            for t in terms:
                rows.append({
                    "URL": url,
                    "Titre": title,
                    "Cluster": cluster,
                    "Priorité": 1 if rank <= int(per_page_terms*0.4) else (2 if rank <= int(per_page_terms*0.8) else 3),
                    "Type": typ,
                    "Terme/Expression": t,
                    "Section suggérée": guess_section(t),
                    "Ancre candidate": anchors[rank % len(anchors)] if anchors else "",
                    "Note": ""
                })
                rank += 1

    return pd.DataFrame(rows)

def export_enriched_csv(df: pd.DataFrame, path="exports/briefs_enriched.csv"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
