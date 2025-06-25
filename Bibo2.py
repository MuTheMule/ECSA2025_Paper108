import pandas as pd
import bibtexparser

# --- Load Semantic Scholar + CrossRef merged metadata ---
sem_cr_df = pd.read_csv("C:/Users/PC/.spyder-py3/Bib/semantic_crossref.csv")  # Already includes source column

# --- Load IEEE ---
ieee_df = pd.read_csv("C:/Users/PC/.spyder-py3/Bib/ieeexplore.csv")
ieee_normalized = pd.DataFrame({
    'title': ieee_df['Document Title'],
    'authors': ieee_df['Authors'],
    'year': ieee_df['Publication Year'],
    'venue': ieee_df['Publication Title'],
    'doi': ieee_df['DOI'],
    'abstract': ieee_df['Abstract'],
    'keywords': ieee_df['Author Keywords'],
    'citation_count': ieee_df['Article Citation Count'],
    'source': 'IEEE'
})

# --- Load ACM BibTeX ---
with open("C:/Users/PC/.spyder-py3/Bib/acm.bib", encoding='utf-8') as bibtex_file:
    acm_bib = bibtexparser.load(bibtex_file)
acm_df = pd.DataFrame(acm_bib.entries)

# Safely combine possible venue fields
venue = acm_df.get('booktitle')
if 'series' in acm_df.columns:
    venue = venue.combine_first(acm_df['series'])
if 'publisher' in acm_df.columns:
    venue = venue.combine_first(acm_df['publisher'])

acm_normalized = pd.DataFrame({
    'title': acm_df.get('title'),
    'authors': acm_df.get('author'),
    'year': acm_df.get('year'),
    'venue': venue,
    'doi': acm_df.get('doi') if 'doi' in acm_df.columns else None,
    'abstract': acm_df.get('abstract'),
    'keywords': acm_df.get('keywords'),
    'citation_count': None,  # ACM does not provide it
    'source': 'ACM'
})

# --- Normalize Semantic Scholar + CrossRef ---
sem_cr_normalized = pd.DataFrame({
    'title': sem_cr_df['title'],
    'authors': sem_cr_df['authors'],
    'year': sem_cr_df['year'],
    'venue': sem_cr_df['venue'],
    'doi': sem_cr_df['doi'],
    'abstract': sem_cr_df['abstract'],
    'keywords': None,  # Add if available
    'citation_count': sem_cr_df['citation_count'],
    'source': sem_cr_df['source']
})

# --- Merge All Sources ---
all_sources_df = pd.concat([ieee_normalized, acm_normalized, sem_cr_normalized], ignore_index=True)

# Normalize for merging
all_sources_df['doi_norm'] = all_sources_df['doi'].str.lower().str.strip()
all_sources_df['title_norm'] = all_sources_df['title'].str.lower().str.strip()

# --- Consolidate duplicates using priority merging ---
def pick_first_non_null(values):
    for v in values:
        if pd.notna(v) and v != '':
            return v
    return None

# Group by DOI (if available), fallback to title
group_key = all_sources_df['doi_norm'].combine_first(all_sources_df['title_norm'])

# --- Identify duplicates before merging ---
all_sources_df['group_key'] = group_key
dup_flags = all_sources_df.duplicated(subset='group_key', keep=False)
all_sources_df['is_duplicate'] = dup_flags

# --- Count duplicates and non-duplicates per source ---
print("Duplicate and Non-Duplicate Counts per Source:")
source_counts = all_sources_df.groupby(['source', 'is_duplicate']).size().unstack(fill_value=0)
source_counts.columns = ['Non-Duplicates', 'Duplicates']
print(source_counts)

# Merge within duplicates
merged = all_sources_df.groupby(group_key).agg({
    'title': pick_first_non_null,
    'authors': pick_first_non_null,
    'year': pick_first_non_null,
    'venue': pick_first_non_null,
    'doi': pick_first_non_null,
    'abstract': pick_first_non_null,
    'keywords': pick_first_non_null,
    'citation_count': pick_first_non_null,
    'source': lambda x: ', '.join(sorted(set(x)))  # Track all sources
}).reset_index(drop=True)

# Save to CSV
merged.to_csv("unified_metadata.csv", index=False)
print(f"Merged dataset saved with {len(merged)} entries.")