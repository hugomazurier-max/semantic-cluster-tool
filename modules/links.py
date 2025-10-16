import pandas as pd
import numpy as np

def suggest_links(analysis, clusters_df, sim, cfg):
    urls = clusters_df["url"].tolist()
    idx = {u:i for i,u in enumerate(urls)}
    rows = []
    for i,u in enumerate(urls):
        c = clusters_df.iloc[i]["cluster"]
        # tri par similarité desc
        sims = [(urls[j], float(sim[i,j])) for j in range(len(urls)) if j!=i]
        sims.sort(key=lambda x: x[1], reverse=True)
        # intra-cluster puis cross
        intra = [ (v,s) for v,s in sims if clusters_df.iloc[idx[v]]["cluster"]==c ][:cfg["linking"]["intra_cluster_topk"]]
        cross = [ (v,s) for v,s in sims if clusters_df.iloc[idx[v]]["cluster"]!=c ][:cfg["linking"]["cross_cluster_topk"]]
        for v,s in intra+cross:
            rows.append({
                "source_url": u,
                "target_url": v,
                "similarité": round(s,3),
                "priorité": 1 if clusters_df.iloc[idx[v]]["cluster"]==c else 2
            })
    return pd.DataFrame(rows)
