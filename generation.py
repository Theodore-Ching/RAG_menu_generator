import numpy as np
import pandas as pd
import random
import os, json, time, ast, pickle
from paths import DATA_DIR, KB_DIR, BM25_PATH, CORPUS_PATH

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Groq client setup
# ------------------------------------------------------------------------------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# query rewriting
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# instead of sending the raw natural-language request directly to BM25, we first ask the LLM to decompose it into 3 focused keyword search strings

_REWRITE_SYSTEM = (
    "You are a recipe search assistant. Your job is to decompose a menu request "
    "into focused keyword search strings for a recipe database. "
    "Respond ONLY with a valid JSON array of exactly 3 strings."
)

_REWRITE_FEW_SHOT = """\
Examples:

Request: "3-course Italian dinner for a party"
Output: ["Italian soup antipasto starter", "Italian pasta main course spaghetti", "Italian tiramisu dessert cake"]

Request: "Suggest a 3-course dinner menu: a soup starter, a chicken main dish, and a cake for dessert."
Output: ["soup starter broth vegetable", "chicken main course roast baked", "cake dessert chocolate vanilla"]

Request: "Create a low-fat, vegetarian 3-course dinner (soup, main, dessert). Avoid using cheese or cream."
Output: ["vegetable soup low-fat starter", "vegetarian main lentil tofu stir-fry", "fruit sorbet low-fat dessert"]
"""

def rewrite_queries_for_courses(user_request: str) -> list:
    """
    Use the LLM to decompose a menu request into 3 focused BM25 search strings.
    Returns a list of 3 strings: [starter_query, main_query, dessert_query].
    Falls back to naive keyword extraction if the LLM response cannot be parsed.
    """
    prompt = (
        f"{_REWRITE_FEW_SHOT}\n"
        f"Request: \"{user_request}\"\n"
        "Output:"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system",  "content": _REWRITE_SYSTEM},
            {"role": "user",    "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()

    # parse JSON array from response
    try:
        # find first [...] block in case the model adds extra text
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        queries = json.loads(match.group(0) if match else raw)
        assert isinstance(queries, list) and len(queries) == 3
        return [str(q) for q in queries]
    except Exception:
        # fallback: return the raw request split into 3 identical queries
        print(f"  [warn] Could not parse LLM rewrite, using raw query. LLM said: {raw!r}")
        return [user_request] * 3

# quick test
test_q = "Plan a 3-course Italian dinner (starter, pasta main, dessert) for a dinner party."
rewritten = rewrite_queries_for_courses(test_q)
courses = ["Starter", "Main", "Dessert"]
print("Query rewriting test:")
for course, q in zip(courses, rewritten):
    print(f"  {course:8s}: {q}")