import numpy as np
import pandas as pd
import random
import os, json, time, ast, pickle
from paths import DATA_DIR, KB_DIR, BM25_PATH, CORPUS_PATH

# setup Groq client
try:
    from groq import Groq
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'groq', '-q'])
    from groq import Groq

import os, json, re

from dotenv import load_dotenv
load_dotenv()  # loads .env from the current directory

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
assert GROQ_API_KEY, (
    "GROQ_API_KEY environment variable is not set.\n"
    "Get a free key at https://console.groq.com, then run:\n"
    "  import os; os.environ['GROQ_API_KEY'] = 'gsk_...'"
)

client    = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "llama-3.3-70b-versatile"

print(f"Groq client ready. Model: {LLM_MODEL}")