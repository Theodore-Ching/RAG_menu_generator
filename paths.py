# paths.py
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'cooking_dataset'
KB_DIR = PROJECT_DIR / "knowledge_base"
BM25_PATH = KB_DIR / "bm25_index.pkl"
CORPUS_PATH = KB_DIR / "corpus.parquet"