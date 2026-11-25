import numpy as np
import pandas as pd
import random
import os, json, time, ast, pickle

# set seeds for full reproducibility
np.random.seed(123)
random.seed(123)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# knowledge base construction
# ------------------------------------------------------------------------------------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'rank-bm25', '-q'])
    from rank_bm25 import BM25Okapi

# paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'cooking_dataset')
KB_DIR   = os.path.join(BASE_DIR, 'knowledge_base')
os.makedirs(KB_DIR, exist_ok=True)

print("Setup complete. rank-bm25 ready.")
print(f"DATA_DIR : {DATA_DIR}")
print(f"KB_DIR   : {KB_DIR}")

# load full dataset (train + dev + test)
# index the full corpus
dfs = []
for split in ['train', 'dev', 'test']:
    path = f'{DATA_DIR}/{split}.csv'
    df = pd.read_csv(path)
    df['_split'] = split
    dfs.append(df)
    print(f"  Loaded {split:5s}: {len(df):,} rows")

corpus_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal recipes in KB corpus: {len(corpus_df):,}")

# build document representations
# document = Title + all ingredients + all recipe steps
def recipe_to_doc(row):
    title = str(row.get('Title', '') or '')

    try:
        ingredients = ' '.join(ast.literal_eval(row['Ingredients']))
    except Exception:
        ingredients = str(row['Ingredients'])
    try:
        steps = ' '.join(ast.literal_eval(row['Recipe']))
    except Exception:
        steps = str(row['Recipe'])

    return f"{title} {ingredients} {steps}"

corpus_df['document'] = corpus_df.apply(recipe_to_doc, axis=1)

print("Sample document (first 300 chars):")
print(corpus_df['document'].iloc[0][:300])