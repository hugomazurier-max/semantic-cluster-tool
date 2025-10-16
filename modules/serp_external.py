import os, time, requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup

def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    return " ".join(soup.get_text(" ").split())

# --- GOOGLE PROGRAMMABLE SEARCH ---
def google_search(query: str, topn: int = 5) -> List[Dict[str, Any]]:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    cx = os.getenv("GOOGLE_CSE_ID", "")
    if not api_key or not cx:
        raise RuntimeError("Clés Google manquantes : définis GOOGLE_API_KEY et GOOGLE_CSE_ID dans .env")

    # l’API renvoie jusqu’à 10 résultats par requête; on gère la pagination 'start'
    items: List[Dict[str, Any]] = []
    fetched = 0
    start = 1
    while fetched < topn:
        num = min(10, topn - fetched)
        params = {"key": api_key, "cx": cx, "q": query, "num": num, "start": start}
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        for it in data.get("items", []):
            items.append({
                "title": it.get("title", ""),
                "url": it.get("link", ""),
                "snippet": it.get("snippet", "")
            })
        if not data.get("items"):
            break
        fetched = len(items)
        start += num
        time.sleep(0.3)
    return items

# --- BING (déjà présent) ---
def bing_search(query: str, topn: int = 5) -> List[Dict[str, Any]]:
    key = os.getenv("BING_SEARCH_KEY", "")
    if not key:
        raise RuntimeError("BING_SEARCH_KEY manquant dans .env")
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {"q": query, "count": topn, "textDecorations": False, "mkt": "fr-FR"}
    headers = {"Ocp-Apim-Subscription-Key": key}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    items = []
    for w in data.get("webPages", {}).get("value", []):
        items.append({"title": w.get("name",""), "url": w.get("url",""), "snippet": w.get("snippet","")})
    return items

def fetch_pages(urls: List[str]) -> List[Dict[str, Any]]:
    out = []
    for u in urls:
        try:
            r = requests.get(u, timeout=15, headers={"User-Agent": "SemanticClusterBot/0.1"})
            if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
                out.append({"url": u, "text": _clean_text(r.text)})
                time.sleep(1.0)
        except Exception:
            continue
    return out
