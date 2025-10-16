import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re, itertools, os

def load_spacy(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(f"Modèle spaCy '{model_name}' non installé. Lance: python -m spacy download {model_name}")

def tokenize_lemma(nlp, text):
    doc = nlp(text)
    toks = [t.lemma_.lower() for t in doc if not (t.is_stop or t.is_punct or t.like_num or t.is_space)]
    return toks

def extract_ents(nlp, text):
    doc = nlp(text)
    ents = [(e.text, e.label_) for e in doc.ents]
    return ents

def get_ngrams(tokens, n=2):
    return list(zip(*[tokens[i:] for i in range(n)]))

def analyze_corpus(docs, cfg):
    nlp = load_spacy(cfg["nlp"]["spacy_model"])
    pages = []
    all_tokens = []

    for d in docs:
        toks = tokenize_lemma(nlp, d["text"])
        pages.append({"url": d["url"], "title": d.get("title",""), "tokens": toks, "text": d["text"]})
        all_tokens.append(toks)

    # TF-IDF
    texts = [" ".join(p["tokens"]) for p in pages]
    tfidf_vec = TfidfVectorizer(max_features=cfg["nlp"]["max_features_tfidf"], ngram_range=tuple(cfg["nlp"]["ngram_range"]))
    X = tfidf_vec.fit_transform(texts)
    vocab = tfidf_vec.get_feature_names_out()

    # Cooccurrences (bigrams/trigrams)
    top_ngrams = []
    for p in pages:
        bigrams = get_ngrams(p["tokens"], 2) + get_ngrams(p["tokens"], 3)
        joined = [" ".join(bg) for bg in bigrams]
        cnt = Counter(joined)
        commons = cnt.most_common(cfg["nlp"]["top_ngrams"])
        top_ngrams.append(commons)

    # NER
    ents_per_page = [extract_ents(nlp, p["text"]) for p in pages]

    pages_df = pd.DataFrame([{"url": p["url"], "title": p["title"]} for p in pages])
    return {
        "pages": pages,
        "pages_df": pages_df,
        "tfidf_vec": tfidf_vec,
        "tfidf_X": X,
        "vocab": vocab,
        "top_ngrams": top_ngrams,
        "ents": ents_per_page,
    }
