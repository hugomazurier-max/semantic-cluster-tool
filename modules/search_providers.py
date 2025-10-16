import os

def web_search_note():
    have = []
    if os.getenv("BING_SEARCH_KEY"): have.append("Bing")
    if os.getenv("SERPAPI_KEY"): have.append("SerpAPI")
    if have:
        return "Recherche Web externe activable : " + ", ".join(have)
    return "Astuce : ajoute une clé dans .env (BING_SEARCH_KEY ou SERPAPI_KEY) pour activer la recherche Web externe (forums, sites)."
