import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
from dotenv import load_dotenv
from modules.crawl import crawl_from_input
import modules.analyze as analyze
from modules.cluster import cluster_pages
from modules.links import suggest_links
from modules.briefs import generate_briefs, export_briefs_csv
from modules.search_providers import web_search_note

# --- CACHE : évite de relancer les gros calculs à chaque interaction ---
@st.cache_data(show_spinner=False)
def _cache_crawl(text_input, input_mode, cfg):
    return crawl_from_input(text_input, input_mode, cfg)

@st.cache_data(show_spinner=False)
def _cache_analyze(docs, cfg):
    return analyze.analyze_corpus(docs, cfg)

@st.cache_data(show_spinner=False)
def _cache_cluster(_analysis, cfg):  # <-- le underscore est crucial ici
    return cluster_pages(_analysis, cfg)

# Configuration de la page Streamlit
st.set_page_config(page_title="Semantic Cluster Tool", layout="wide")
load_dotenv()

# --- STATE: initialisation des clés ---
for key, default in [
    ("docs", None),
    ("analysis", None),
    ("clusters", None),
    ("sim", None),
    ("links_df", None),
    ("briefs_df", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🧩 Semantic Cluster Tool — Starter")
st.caption("Crawl • TF-IDF • Cooccurrences • NER • Similarités • Clusters • Maillage • Briefs")

# Lecture du fichier config.yaml
cfg_path = Path("config.yaml")
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

# Barre latérale avec la config
with st.sidebar:
    st.header("Configuration")
    st.code(Path("config.yaml").read_text(), language="yaml")
    st.info(web_search_note())

# Étape 1 — Ingestion
st.subheader("1) Ingestion")
input_mode = st.radio("Source", ["Sitemap URL", "Liste d’URLs (une par ligne)"])
text_input = st.text_area("Saisis ton sitemap ou tes URLs :", height=150)
start = st.button("Lancer le crawl")

if start and text_input.strip():
    with st.spinner("Crawl en cours…"):
        docs = _cache_crawl(text_input, input_mode, cfg)
    st.session_state.docs = docs

if st.session_state.docs:
    docs = st.session_state.docs
    st.subheader("1) Ingestion")
    st.success(f"{len(docs)} pages crawlé(es).")
    st.dataframe(pd.DataFrame([{"url": d['url'], "title": d.get('title', '')} for d in docs]))

    # Étape 2 — Analyse NLP
    if st.session_state.analysis is None:
        st.subheader("2) Analyse NLP")
        with st.spinner("Analyse TF-IDF / cooccurrences / NER / embeddings…"):
            st.session_state.analysis = _cache_analyze(docs, cfg)

    analysis = st.session_state.analysis
    st.success("Analyse terminée.")
    st.dataframe(analysis["pages_df"].head(20))

    # Étape 3 — Clustering & similarités
    if st.session_state.clusters is None or st.session_state.sim is None:
        st.subheader("3) Clustering & similarités")
        st.session_state.clusters, st.session_state.sim = _cache_cluster(analysis, cfg)

    clusters, sim = st.session_state.clusters, st.session_state.sim
    st.subheader("3) Clustering & similarités")
    st.success(f"{clusters['cluster'].nunique()} clusters trouvés.")
    st.write("Répartition par cluster :")
    st.dataframe(clusters['cluster'].value_counts().rename_axis('cluster').reset_index(name='pages'))
    st.dataframe(clusters.head(30))

    # --- Descripteurs par cluster (top termes TF-IDF) ---
    def top_terms_by_cluster(analysis, clusters, topn=10):
        # moyenne TF-IDF par cluster
        X = analysis["tfidf_X"]
        vocab = analysis["vocab"]
        pages = analysis["pages"]
        import numpy as np, pandas as pd

        url_to_idx = {p["url"]: i for i, p in enumerate(pages)}
        rows = []
        for c_id, sub in clusters.groupby("cluster"):
            idxs = [url_to_idx[u] for u in sub["url"].tolist() if u in url_to_idx]
            if not idxs:
                continue
            vec = X[idxs].mean(axis=0).A1
            top_idx = vec.argsort()[::-1][:topn]
            terms = [vocab[i] for i in top_idx if vec[i] > 0]
            rows.append({"cluster": int(c_id), "top_terms": ", ".join(terms)})
        return pd.DataFrame(rows)

    st.write("Descripteurs de clusters (top termes) :")
    st.dataframe(top_terms_by_cluster(analysis, clusters, topn=10))

    # Étape 4 — Liens internes & ancres
    st.subheader("4) Liens internes & ancres")
    if st.session_state.links_df is None:
        st.session_state.links_df = suggest_links(analysis, clusters, sim, cfg)

        # Fallback pour target_title s'il manque
        links_df_tmp = st.session_state.links_df.copy()
        if ("target_title" not in links_df_tmp.columns) or (links_df_tmp["target_title"].isna().all()):
            def slug_to_title(u):
                try:
                    slug = str(u).strip("/").split("/")[-1]
                    return slug.replace("-", " ").title()
                except Exception:
                    return str(u)
            if "target_url" in links_df_tmp.columns:
                links_df_tmp["target_title"] = links_df_tmp["target_url"].apply(slug_to_title)
        st.session_state.links_df = links_df_tmp

    links_df = st.session_state.links_df
    st.dataframe(links_df.head(40))
    st.download_button(
        "Télécharger la matrice de liens (CSV)",
        links_df.to_csv(index=False).encode("utf-8"),
        "matrice_liens.csv"
    )

    # Étape 5 — Briefs lexicaux
    st.subheader("5) Briefs lexicaux")
    if st.session_state.briefs_df is None:
        st.session_state.briefs_df = generate_briefs(analysis, clusters, cfg)

    briefs_df = st.session_state.briefs_df
    st.dataframe(briefs_df.head(40))
    st.download_button(
        "Télécharger les briefs (CSV)",
        briefs_df.to_csv(index=False).encode("utf-8"),
        "briefs_lexicaux.csv"
    )
    export_briefs_csv(briefs_df)
    st.success("Exports CSV générés dans le dossier ./exports")

    # 6) Briefs enrichis (beaucoup plus fournis)
    st.subheader("6) Briefs enrichis (TERMES + ENTITÉS + COOC + QUESTIONS + ANCRES)")
    per_terms = st.slider("Termes à générer par page", 40, 120, 80, 10)
    if st.button("Générer les briefs enrichis"):
        from modules.enrich import enrich_page, export_enriched_csv
        enriched = enrich_page(analysis, clusters, links_df, per_page_terms=per_terms)

        # Remplir automatiquement la colonne "Note" avec section + ancre
        if "Note" in enriched.columns and "Ancre candidate" in enriched.columns:
            enriched["Note"] = enriched.apply(
                lambda r: f"Intégrer en {r['Section suggérée']} avec ancre « {r['Ancre candidate']} »" if r["Ancre candidate"] else "",
                axis=1
            )

        st.dataframe(enriched.head(200))
        st.download_button(
            "Télécharger briefs enrichis (CSV)",
            enriched.to_csv(index=False).encode("utf-8"),
            "briefs_enriched.csv"
        )
        export_enriched_csv(enriched)
        st.success("Briefs enrichis exportés dans ./exports/briefs_enriched.csv")

        # --- Onglets PRO (doivent être DANS le if) ---
    tab1, tab2, tab3 = st.tabs(["Briefs PRO", "Cocons (Mots-clés)", "Recherche externe"])

    with tab1:
        st.markdown("### Briefs PRO (cibles / 1000 mots + score de couverture)")
        per_page_terms = st.slider("Nombre de termes par page", 20, 60, 40, 5)
        target_len = st.slider("Longueur de référence (mots)", 800, 2000, 1200, 100)
        if st.button("Générer les Briefs PRO"):
            from modules.briefs_pro import generate_briefs_pro, export_briefs_pro_csv
            briefs_pro = generate_briefs_pro(
                analysis, clusters,
                target_len_words=target_len,
                per_page_terms=per_page_terms
            )
            st.dataframe(briefs_pro.head(80))
            st.download_button(
                "Télécharger briefs PRO (CSV)",
                briefs_pro.to_csv(index=False).encode("utf-8"),
                "briefs_pro.csv"
            )
            export_briefs_pro_csv(briefs_pro)
            st.success("Briefs PRO exportés dans ./exports/briefs_pro.csv")

    with tab2:
        st.markdown("### Générateur de cocons à partir d'une liste de mots-clés")
        kws_text = st.text_area("Colle une liste de mots-clés (un par ligne)", height=200)
        n_clusters = st.number_input(
            "Nombre de clusters (laisser 0 pour auto)",
            min_value=0, max_value=50, value=0, step=1
        )
        if st.button("Clusteriser les mots-clés"):
            from modules.keywords import cluster_keywords, export_cocons_to_csv
            kws = [k.strip() for k in kws_text.splitlines() if k.strip()]
            if kws:
                cocons_df = cluster_keywords(
                    kws,
                    n_clusters if n_clusters > 0 else None,
                    model_name=cfg["similarity"]["model_name"]
                )
                st.dataframe(cocons_df.head(100))
                st.download_button(
                    "Télécharger cocons (CSV)",
                    cocons_df.to_csv(index=False).encode("utf-8"),
                    "cocons_keywords.csv"
                )
                export_cocons_to_csv(cocons_df)
                st.success("Cocons exportés dans ./exports/cocons_keywords.csv")
            else:
                st.warning("Ajoute au moins un mot-clé.")

    with tab3:
        st.markdown("### Recherche externe (SERP) — optionnelle")

        provider = st.radio(
            "Provider",
            ["google", "bing"],
            index=0,
            horizontal=True,
            key="serp_provider"
        )

        if provider == "google":
            st.info("Active Google: ajoute GOOGLE_API_KEY et GOOGLE_CSE_ID dans .env")
        else:
            st.info("Active Bing: ajoute BING_SEARCH_KEY dans .env")

        query = st.text_input("Requête (ex: peinture acrylique débutant)", key="serp_query")
        topn = st.slider("Nombre de résultats", 1, 10, 5, key="serp_topn")

        if st.button("Rechercher", key="serp_go"):
            if not query:
                st.warning("Saisis une requête.")
            else:
                try:
                    from modules.serp_external import google_search, bing_search, fetch_pages

                    if provider == "google":
                        results = google_search(query, topn=topn)
                    else:
                        results = bing_search(query, topn=topn)

                    if results:
                        st.dataframe(results)
                        urls = [r["url"] for r in results]
                        pages = fetch_pages(urls)
                        st.write(f"Pages récupérées: {len(pages)}")
                    else:
                        st.info("Aucun résultat renvoyé par l'API.")

                except Exception as e:
                    st.error(str(e))



