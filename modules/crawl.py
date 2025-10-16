import requests, time, re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import trafilatura
import readability
from readability import Document as ReadabilityDoc
import urllib.robotparser as robotparser
from .utils import same_domain, clean_text

def can_fetch(url, ua):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(ua, url)
    except Exception:
        return True  # si robots inaccessible, on reste prudent via throttling

def extract_readable(url, html):
    try:
        doc = ReadabilityDoc(html)
        title = doc.short_title()
        content_html = doc.summary()
        text = trafilatura.extract(content_html, include_comments=False, include_tables=False) or ""
        return title, text
    except Exception:
        return "", ""

def crawl_from_input(text_input: str, input_mode: str, cfg: dict):
    ua = cfg["crawl"]["user_agent"]
    max_pages = cfg["crawl"]["max_pages"]
    max_depth = cfg["crawl"]["max_depth"]
    same_only = cfg["crawl"]["same_domain_only"]
    delay = cfg["crawl"]["request_delay_seconds"]

    urls = []
    if input_mode == "Sitemap URL":
        urls = [text_input.strip()]
    else:
        urls = [u.strip() for u in text_input.splitlines() if u.strip()]

    # Si sitemap, extraire URLs
    seeds = []
    for u in urls:
        if u.endswith(".xml"):
            try:
                r = requests.get(u, timeout=15, headers={"User-Agent": ua})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "xml")
                locs = [loc.text.strip() for loc in soup.find_all("loc")]
                seeds.extend(locs)
            except Exception:
                pass
        else:
            seeds.append(u)

    seen = set()
    queue = [(s, 0) for s in seeds]
    docs = []

    base_domain = urlparse(seeds[0]).netloc if seeds else ""

    while queue and len(docs) < max_pages:
        url, depth = queue.pop(0)
        if url in seen: 
            continue
        seen.add(url)

        if same_only and base_domain and not same_domain(url, f"https://{base_domain}"):
            continue

        if not can_fetch(url, ua):
            continue

        try:
            time.sleep(delay)
            resp = requests.get(url, timeout=20, headers={"User-Agent": ua})
            if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type",""):
                continue
            title, text = extract_readable(url, resp.text)
            if not text:
                continue

            docs.append({"url": url, "title": title, "text": clean_text(text)})
            if depth < max_depth:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("#"): 
                        continue
                    nxt = urljoin(url, href)
                    queue.append((nxt, depth+1))
        except Exception:
            continue

    return docs
