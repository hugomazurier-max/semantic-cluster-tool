from urllib.parse import urlparse

def same_domain(u1, u2):
    d1 = urlparse(u1).netloc
    d2 = urlparse(u2).netloc
    return d1.lower() == d2.lower()

def clean_text(s: str) -> str:
    return " ".join(s.split())
