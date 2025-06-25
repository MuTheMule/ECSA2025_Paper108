import requests
import pandas as pd
import time
import string
import re

# ---- CONFIGURATION ---- #
query_keywords = [
    "system of systems", "sos",
    "internet of things", "iot", "cyber physical system", "cps"
    #comment the following keywords for a broader search, uncomment for the specific search
    "resilience", "adaptability", "dynamic reconfiguration"
]
query_string = " ".join(query_keywords)

# Output file
output_csv = "semantic_crossref.csv"

# ---- SEMANTIC SCHOLAR API ---- #
def query_semantic_scholar(query, limit=100, max_results=500):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    results = []

    for offset in range(0, max_results, limit):
        params = {
            "query": query,
            "fields": "title,abstract,authors,year,venue,externalIds,citationCount",
            "limit": limit,
            "offset": offset
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json().get('data', [])
        except Exception as e:
            print(f"[Semantic Scholar] Failed at offset {offset}: {e}")
            break  # Stop querying, but return what we got so far

        for paper in data:
            results.append({
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": ", ".join(a["name"] for a in paper.get("authors", [])),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "doi": paper.get("externalIds", {}).get("DOI"),
                "source": "Semantic Scholar",
                "citation_count": paper.get("citationCount")
            })

        print(f"Retrieved {len(data)} papers from offset {offset}")
        time.sleep(3)

    return results

# ---- CROSSREF API ---- #
def query_crossref(query, rows=100):
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": rows,
        "filter": "from-pub-date:2010-01-01,until-pub-date:2025-05-31"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"[CrossRef] Query failed: {e}")
        return []  # Skip CrossRef and move on

    results = []
    for item in items:
        results.append({
            "title": item.get("title", [""])[0],
            "abstract": item.get("abstract", ""),
            "authors": ", ".join(f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])) if "author" in item else "",
            "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
            "venue": item.get("container-title", [""])[0],
            "doi": item.get("DOI"),
            "source": "CrossRef",
            "citation_count": None
        })

    return results

# ---- RUN QUERIES ---- #
print("Querying Semantic Scholar...")
sem_results = query_semantic_scholar(query="system of systems iot cps", limit=100)
print(f"Retrieved {len(sem_results)} results from Semantic Scholar.")

print("Querying CrossRef...")
cr_results = query_crossref(query_string, rows=100)
print(f"Retrieved {len(cr_results)} results from CrossRef.")

# ---- COMBINE AND SAVE ---- #
all_results = sem_results + cr_results
df = pd.DataFrame(all_results)
df.drop_duplicates(subset=["title", "doi"], inplace=True)
df.to_csv(output_csv, index=False)

print(f"\nSaved {len(df)} merged results to: {output_csv}")

def clean_doi(doi):
    if pd.isna(doi):
        return None
    doi = doi.strip().lower()
    doi = re.sub(r'(https?://)?(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:\s*', '', doi)
    return doi.strip()


df['clean_doi'] = df['doi'].apply(clean_doi)

# --- Step 2: Normalize Titles ---
def clean_title(title):
    if pd.isna(title):
        return None
    title = title.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    return title

df['clean_title'] = df['title'].apply(clean_title)
df.head()
