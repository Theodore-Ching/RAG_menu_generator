# retrieval function (with stopword filtering)
# BM25 IDF assigns very high scores to words that are rare in a recipe corpus.
# natural-language query prefixes like "Give me some..." contain words that
# almost never appear in recipe text, so they get huge IDF boosts and drown out
# the actual content keywords (e.g. "chicken"). filtering these stopwords first
# ensures BM25 scores only on semantically meaningful query terms.

import re as _re
import numpy as np
import pandas as pd
import random
import os, json, time, ast, pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from paths import DATA_DIR, KB_DIR, BM25_PATH, CORPUS_PATH

# load index and corpus from disk
with open(BM25_PATH, 'rb') as f: bm25 = pickle.load(f)
corpus_df = pd.read_parquet(CORPUS_PATH)


_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "give", "recipe", "recipes", "meal", "meals", "dish", "dishes",
    'food', 'please', 'want', 'suggest', 'create', 'plan', 'prepare',
    'using', 'use', 'without', 'avoid', 'menu', 'course', 'courses'
}

def retrieve_top_k(query: str, k: int = 5) -> pd.DataFrame:
    """
    Query the BM25 index and return the top-k most relevant recipes.
    Stopwords are stripped from the query before scoring so that natural-
    language prefixes ("Give me some...") don't corrupt the BM25 ranking.
    Returns a DataFrame with columns: Title, Ingredients, Recipe, bm25_score.
    """
    raw_tokens = _re.sub(r"[^\w\s]", "", query.lower()).split()
    tokens = [t for t in raw_tokens if t not in _STOPWORDS]
    if not tokens:          # fallback: keep everything if all words were stopwords
        tokens = raw_tokens
    scores = bm25.get_scores(tokens)
    top_k_idx = np.argsort(scores)[::-1][:k]
    results = corpus_df.iloc[top_k_idx][['Title', 'Ingredients', 'Recipe']].copy()
    results['bm25_score'] = scores[top_k_idx]
    return results.reset_index(drop=True)

# measure retrieval latency
sample_query = "Give me some chicken recipes."
_ = retrieve_top_k(sample_query)       # warm-up
t0 = time.time()
sample_results = retrieve_top_k(sample_query)
latency_ms = (time.time() - t0) * 1000

print(f"Retrieval latency (sample)   : {latency_ms:.1f} ms")
print()
print(f"Top-5 results for: '{sample_query}'")
print(sample_results[['Title', 'bm25_score']].to_string())