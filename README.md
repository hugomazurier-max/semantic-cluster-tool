# Semantic Cluster Tool — Starter Kit

Un outil local (Streamlit) pour :
- Crawler des pages (URL/Sitemap), respecter `robots.txt`, et extraire le contenu propre.
- Calculer TF‑IDF, cooccurrences, entités nommées (NER), similarités (embeddings).
- Regrouper en clusters, suggérer du maillage interne + ancres.
- Générer des *briefs lexicaux* par page (priorités, sections, ancres).
- (Option) Explorer le Web via une API de recherche si tu ajoutes une clé (Bing, SerpAPI).

## 1) Installation rapide

### Windows / macOS / Linux
1. Installe Python 3.10+ : https://www.python.org/downloads/
2. Ouvre un terminal dans ce dossier et crée un environnement :
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
   python -m spacy download fr_core_news_lg
   ```

> Remarque: la première exécution téléchargera un modèle d'embeddings `sentence-transformers` (connexion internet requise une seule fois).

## 2) Lancer l'app
```bash
streamlit run app.py
```
Ouvre ensuite l’URL locale affichée (typiquement http://localhost:8501).

## 3) Flux de travail
1. **Ingestion** : colle un sitemap ou une liste d’URLs. Option *BFS interne* pour découvrir de nouvelles pages dans le même domaine.
2. **Analyse** : TF‑IDF, cooccurrences, NER, embeddings + similarités, clustering.
3. **Cocons & maillage** : suggestions de liens internes (priorité intra-cluster), ancres candidates.
4. **Briefs lexicaux** : par page (priorités, sections, ancres, notes). Export CSV.
5. **Exploration Web (option)** : fournis une clé d’API de recherche dans `.env` pour élargir à des forums/sites externes.

## 4) Respect & conformité
- **Respecte `robots.txt`**, limites de taux (throttle), et conditions d’utilisation des sites.
- Utilise ce projet à des fins d’analyse éditoriale légitime. Tu es responsable de l’usage que tu en fais.

## 5) Personnalisation rapide
- Ajuste `config.yaml` (seuils de similarité, top‑k liens, profondeur crawl).
- Modifie `keywords_stopwords.txt` pour tes stopwords métier.
- Ajoute des *providers* de recherche dans `search_providers.py` (Bing, SerpAPI…).

Bon build !
