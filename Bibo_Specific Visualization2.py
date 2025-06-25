import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import nltk
import string
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from matplotlib import colors

nltk.download('stopwords')

# ========== LOAD DATA ==========
df = pd.read_csv("C:/Users/PC/.spyder-py3/Bib/Specific/unified_metadata.csv")

# ========== SETUP ==========
# Normalizing DOI and Title for deduplication
df['doi_norm'] = df['doi'].str.lower().str.strip()
df['title_norm'] = df['title'].str.lower().str.strip()

# Use DOI as primary group key, fallback to title
df['group_key'] = df['doi_norm'].fillna(df['title_norm'])

# ========== GRAPH 1: Co-Occurrence of Sources ==========
G = nx.Graph()
source_counts = df['source'].value_counts().to_dict()

for src in df['source'].unique():
    G.add_node(src, size=source_counts[src])

pair_links = defaultdict(int)
grouped = df.groupby('group_key')

for _, group in grouped:
    sources = group['source'].unique()
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            pair = tuple(sorted((sources[i], sources[j])))
            pair_links[pair] += 1

for (src1, src2), weight in pair_links.items():
    G.add_edge(src1, src2, weight=weight)

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=30, k=0.3)
sizes = [G.nodes[n]['size'] * 30 for n in G.nodes]
node_colors = ['#90ee90' if G.nodes[n]['size'] > 1 else '#c0ffc0' for n in G.nodes]
node_border_color = 'darkgreen'
edge_weights = [e[2]['weight'] * 1.5 for e in G.edges(data=True)]

nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, edgecolors=node_border_color, linewidths=2)
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='#4682b4', style='dashed', alpha=0.7, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_labels(G, pos, font_size=24, font_family='Georgia', font_weight='bold')
plt.title("Publication Count & Co-occurrence by Data Source")
plt.axis('off')
plt.tight_layout()
plt.show()

# ========== DEDUPLICATION ==========
duplicates_doi = df[df['doi'].notna()].duplicated(subset='doi', keep='first')
duplicates_title = df[df['doi'].isna()].duplicated(subset='title', keep='first')
duplicates_combined = duplicates_doi | duplicates_title

duplicates_df = df[duplicates_combined]

print("Number of duplicate entries by source:")
if not duplicates_df.empty:
    print(duplicates_df['source'].value_counts())
else:
    print("No duplicate entries found based on DOI or Title.")

duplicates_removed = df[duplicates_combined][["source", "title", "doi"]]

final_df = df[~duplicates_combined].reset_index(drop=True)

final_df['year'] = pd.to_numeric(final_df['year'], errors='coerce')
final_df = final_df[final_df['year'] > 2010].dropna(subset=['year'])

# ========== GRAPH 2: Term Frequency Over Years ==========
# Defining search terms
iot_keywords = ["cyber physical system", "cps"]
sos_keywords = ["system of systems", "sos"]
resilience_keywords = ["resilience", "adaptability"]
all_terms = iot_keywords + sos_keywords + resilience_keywords

def extract_terms(text):
    text = str(text).lower()
    return [kw for kw in all_terms if kw in text]

final_df['search_term'] = final_df['title'] + " " + final_df['abstract'].fillna("")
final_df['matched_terms'] = final_df['search_term'].apply(extract_terms)
final_df = final_df.explode('matched_terms').rename(columns={'matched_terms': 'main_term'})
final_df = final_df[final_df['main_term'].notna() & (final_df['main_term'] != "Other")]

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=final_df,
    x='year',
    y=final_df.groupby(['year', 'main_term'])['title'].transform('count'),
    hue='main_term',
    estimator=None,
    lw=2
)
plt.ylabel("Number of Publications", fontname='Georgia', fontsize=18)
plt.xlabel("Year", fontname='Georgia', fontsize=18)
plt.xticks(fontname='Georgia', fontsize=16)
plt.yticks(fontname='Georgia', fontsize=16)
legend = plt.legend(title="Search Term")
plt.setp(legend.get_title(), fontname='Georgia', fontsize=11)
plt.setp(legend.get_texts(), fontname='Georgia', fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== GRAPH 3: Keyword Co-Occurrence (from Title + Abstract + Keywords) ==========
# Combining 'title', 'abstract', and 'keywords' into a single text field
def combine_text_fields(row):
    components = [
        str(row.get('title', '')).lower(),
        str(row.get('abstract', '')).lower(),
        str(row.get('keywords', '')).lower()
    ]
    return ' '.join(components)

final_df['combined_text'] = final_df.apply(combine_text_fields, axis=1)

# Cleaning and tokenization
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

stemmed_keywords = []

for text in final_df['combined_text'].dropna():
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    # Lowercase, remove stopwords, and applying stemming
    processed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    stemmed_keywords.append(processed)

final_df['stemmed_keywords'] = stemmed_keywords

# Computing keyword frequencies
keyword_freq = Counter()
for kw_list in stemmed_keywords:
    keyword_freq.update(kw_list)

# Keeping only frequently occurring keywords
min_count = 100
valid_keywords = {kw for kw, count in keyword_freq.items() if count >= min_count}

# Computing co-occurrence
co_occurrence = defaultdict(int)
for tokens in stemmed_keywords:
    filtered_kw = [kw for kw in tokens if kw in valid_keywords]
    unique_tokens = set(filtered_kw)
    for token1 in unique_tokens:
        for token2 in unique_tokens:
            if token1 < token2:
                co_occurrence[(token1, token2)] += 1

# Build and visualize graph
G = nx.Graph()

for kw in valid_keywords:
    G.add_node(kw, size=keyword_freq[kw])

for (kw1, kw2), weight in co_occurrence.items():
    G.add_edge(kw1, kw2, weight=weight)

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=20, k=0.4)

node_freqs = [G.nodes[n]['size'] for n in G.nodes]
norm = colors.Normalize(vmin=min(node_freqs), vmax=max(node_freqs), clip=True)
alphas = [0.4 + 0.6 * norm(freq) for freq in node_freqs]

for node, size, alpha in zip(G.nodes, node_freqs, alphas):
    nx.draw_networkx_nodes(G, pos,
        nodelist=[node],
        node_size=size * 20,
        node_color='mediumslateblue',
        alpha=alpha,
        edgecolors='black',
        linewidths=1.2)

nx.draw_networkx_edges(G, pos,
    width=[d['weight'] * 0.2 for (_, _, d) in G.edges(data=True)],
    alpha=0.4,
    edge_color='gray',
    style='solid')

nx.draw_networkx_labels(G, pos, font_size=14, font_family='Georgia', font_weight='ultralight')

plt.axis('off')
plt.tight_layout()
plt.show()

print("\nMost Frequent Keywords:")
most_common_keywords = keyword_freq.most_common(20)
for kw, count in most_common_keywords:
    print(f"{kw}: {count}")

print("\nMost Frequent Keyword Co-occurrences:")
most_common_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:20]
for (kw1, kw2), count in most_common_pairs:
    print(f"{kw1} & {kw2}: {count}")

print("\nMost Cited Papers:")
if 'citation_count' in final_df.columns:
    top_cited = final_df[['title', 'authors', 'citation_count', 'year']].dropna(subset=['citation_count'])
    top_cited['citation_count'] = pd.to_numeric(top_cited['citation_count'], errors='coerce')
    top_cited = top_cited.sort_values(by='citation_count', ascending=False).head(10)

    for idx, row in top_cited.iterrows():
        print(f"\nTitle: {row['title']}\nAuthors: {row['authors']}\nCitations: {int(row['citation_count'])} ({int(row['year'])})")
else:
    print("Citation count not found in dataset.")
    
#Manual seclection of some of the most mentioned keywords
keywords = [
    "secur", "model", "data", "use", "network",
    "approach", "integr", "framework", "technolog", "control"]

frequencies = [995, 893, 777, 706, 661, 632, 615, 530, 502, 501]

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(keywords, frequencies, color='skyblue', edgecolor='black', height=0.5)

# Add value labels on top of each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 25, bar.get_y() + bar.get_height()/2, str(width),
             ha='center', va='bottom', fontsize=14, fontname='Georgia')

# Label and styling
plt.xlabel("Frequency", fontsize=14, fontname='Georgia')
plt.ylabel("Keyword Stem", fontsize=14, fontname='Georgia')
plt.xticks(fontsize=14, fontname='Georgia')
plt.yticks(fontsize=14, fontname='Georgia')
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()