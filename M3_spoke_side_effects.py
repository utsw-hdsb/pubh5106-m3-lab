# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SPOKE Side Effect Explorer
#
# **PUBH 5106 — AI in Health Applications, Module 3 Supplement**
#
# This notebook queries SPOKE's compound-side effect data (sourced from
# SIDER) for any drugs you specify. It shows individual and shared side
# effects, organized by clinical category.
#
# ## Use Case
#
# A patient treated with vincristine and dactinomycin for childhood
# nephroblastoma wants to understand what long-term effects these drugs
# are associated with in the knowledge graph.

# %%
from collections import defaultdict

import requests
import pandas as pd
from IPython.display import display, Markdown

SPOKE_API = "https://spoke.rbvi.ucsf.edu/api/v1"

# %% [markdown]
# ## 1. Configure Your Drugs
#
# Add or remove drugs below. The notebook will query SPOKE for each one.

# %%
# Search SPOKE for drug identifiers
def spoke_search(query, node_type="Compound"):
    """Search SPOKE for a compound by name."""
    r = requests.get(f"{SPOKE_API}/search/{node_type}/{query}", timeout=30)
    r.raise_for_status()
    return r.json()


def spoke_neighborhood(node_type, identifier):
    """Get 1-hop neighborhood from SPOKE."""
    url = f"{SPOKE_API}/neighborhood/{node_type}/identifier/{identifier}"
    r = requests.post(url, json={"cutoff_Specificity_Index": 0}, timeout=60)
    r.raise_for_status()
    return r.json()


# Find identifiers for our drugs
DRUG_QUERIES = ["vincristine", "dactinomycin"]

print("Searching SPOKE for drug identifiers...\n")
drug_search_results = {}
for query in DRUG_QUERIES:
    results = spoke_search(query)
    drug_search_results[query] = results
    print(f"  {query}: {len(results)} entries in SPOKE")
    for r in results:
        print(f"    {r['name']} ({r['identifier']})")

# %% [markdown]
# ### Compound identifier problem
#
# Many drugs exist as multiple compound entries in SPOKE (different salt
# forms, stereoisomers, or naming variants). Side effect data from SIDER
# may only be attached to ONE of these entries. We need to check all
# variants and merge their side effects.

# %%
# For each drug, check ALL variants and keep the one(s) with side effects
drug_ids = {}

for query, results in drug_search_results.items():
    best_name = None
    all_ses = set()
    all_data = []

    for r in results:
        data = spoke_neighborhood("Compound", r["identifier"])
        ses = {item["data"]["properties"].get("name", "?")
               for item in data if item["data"]["neo4j_type"] == "SideEffect"}
        print(f"  {r['name']} ({r['identifier']}): {len(ses)} side effects")

        if ses:
            all_ses.update(ses)
            all_data.extend(data)
            if best_name is None:
                best_name = r["name"]

    if best_name is None:
        best_name = results[0]["name"] if results else query

    drug_ids[best_name] = {
        "side_effects": all_ses,
        "data": all_data,
    }
    print(f"  → Using '{best_name}': {len(all_ses)} side effects (merged)\n")

# %% [markdown]
# ## 2. Side Effect Results

# %%
drug_side_effects = {}

for drug_name, info in drug_ids.items():
    drug_side_effects[drug_name] = sorted(info["side_effects"])
    print(f"{drug_name}: {len(info['side_effects'])} side effects in SPOKE")

# %% [markdown]
# ## 3. Individual Side Effects

# %%
for drug_name, ses in drug_side_effects.items():
    print(f"\n{'='*60}")
    print(f"{drug_name} — {len(ses)} side effects (SIDER)")
    print(f"{'='*60}")
    if ses:
        for se in ses:
            print(f"  {se}")
    else:
        print("  No side effect data in SPOKE for this drug.")
        print("  (SIDER may not have an entry, or the drug may be")
        print("  mapped under a different compound identifier.)")

# %% [markdown]
# ## 4. Shared vs. Unique Side Effects
#
# If a side effect appears with both drugs, a patient receiving the
# combination may have had compounded risk.

# %%
if len(drug_side_effects) >= 2:
    drug_names = list(drug_side_effects.keys())
    sets = {name: set(ses) for name, ses in drug_side_effects.items()}

    # Shared across all drugs
    shared = set.intersection(*sets.values()) if all(sets.values()) else set()

    print(f"=== Shared side effects ({len(shared)}) ===")
    if shared:
        for se in sorted(shared):
            print(f"  {se}")
    else:
        print("  None (one or more drugs have no side effect data)")

    # Unique to each drug
    for name in drug_names:
        unique = sets[name] - set.union(*(s for n, s in sets.items() if n != name)) if sets[name] else set()
        print(f"\n=== Unique to {name} ({len(unique)}) ===")
        for se in sorted(unique):
            print(f"  {se}")
else:
    print("Need at least 2 drugs to compare.")

# %% [markdown]
# ## 5. Side Effects by Clinical Category
#
# SPOKE stores side effects as MedDRA preferred terms. We can loosely
# categorize them by clinical system using keyword matching. This is
# approximate — a formal MedDRA System Organ Class mapping would be
# more precise.

# %%
# Rough clinical categories by keyword
CATEGORIES = {
    "Neurological": ["neuropathy", "neurotox", "ataxia", "paralysis",
                     "paresis", "seizure", "convulsion", "coma",
                     "numbness", "paraesthesia", "nystagmus", "vertigo",
                     "dizziness", "headache", "footdrop", "gait",
                     "myoclonus", "blindness", "optic", "hearing",
                     "deafness", "consciousness", "hallucination",
                     "sensory loss", "cranial nerve", "walking",
                     "peroneal"],
    "Gastrointestinal": ["nausea", "vomiting", "diarrhoea", "diarrhea",
                         "constipation", "stomatitis", "ileus",
                         "abdominal", "intestinal", "anorexia",
                         "appetite", "mouth", "gastrointestinal",
                         "necrosis"],
    "Hematological": ["anaemia", "anemia", "leukopenia", "leuko",
                      "thrombocytopenia", "pancytopenia",
                      "bone marrow", "leukaemia"],
    "Cardiovascular": ["hypertension", "hypotension", "cardiac",
                       "myocardial", "coronary", "acute coronary"],
    "Dermatological": ["rash", "alopecia", "dermatitis",
                       "photosensitivity", "sweating", "hyperhidrosis"],
    "Renal/Urological": ["dysuria", "polyuria", "urinary", "urine",
                         "azotaemia", "urate", "erectile"],
    "Musculoskeletal": ["myalgia", "muscle", "bone pain", "back pain",
                        "pain in extremity", "pain in jaw",
                        "musculoskeletal"],
    "Metabolic/Endocrine": ["hyponatraemia", "hyperuricaemia",
                            "dehydration", "weight", "uric acid",
                            "adrenal", "temperature"],
    "Psychiatric": ["depression", "insomnia", "agitation"],
    "Other": [],  # catch-all
}


def categorize_side_effect(se_name):
    """Assign a side effect to a clinical category."""
    se_lower = se_name.lower()
    for category, keywords in CATEGORIES.items():
        if category == "Other":
            continue
        if any(kw in se_lower for kw in keywords):
            return category
    return "Other"


# Build categorized table for all drugs combined
all_ses = set()
for ses in drug_side_effects.values():
    all_ses.update(ses)

rows = []
for se in sorted(all_ses):
    category = categorize_side_effect(se)
    row = {"Side Effect": se, "Category": category}
    for drug_name in drug_ids:
        row[drug_name] = "X" if se in drug_side_effects.get(drug_name, []) else ""
    rows.append(row)

df = pd.DataFrame(rows)
if not df.empty:
    # Sort by category then name
    df = df.sort_values(["Category", "Side Effect"])

    print(f"{'Category':<25} {'Side Effect':<35}", end="")
    for name in drug_ids:
        print(f" {name:>15}", end="")
    print()
    print("-" * (75 + 16 * len(drug_ids)))

    current_cat = None
    for _, row in df.iterrows():
        if row["Category"] != current_cat:
            current_cat = row["Category"]
            print(f"\n  {current_cat}")
        marks = "  ".join(
            f"{'X':>15}" if row.get(name) == "X" else f"{'':>15}"
            for name in drug_ids
        )
        print(f"    {row['Side Effect']:<33} {marks}")
else:
    print("No side effect data available.")

# %% [markdown]
# ## 6. Summary Table

# %%
if not df.empty:
    # Count by category
    summary_rows = []
    for category in CATEGORIES:
        cat_df = df[df["Category"] == category]
        if cat_df.empty:
            continue
        row = {"Category": category, "Total": len(cat_df)}
        for drug_name in drug_ids:
            row[drug_name] = (cat_df[drug_name] == "X").sum()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    display(summary.set_index("Category"))

    # Total
    print(f"\nTotal unique side effects across all drugs: {len(all_ses)}")
    for drug_name, ses in drug_side_effects.items():
        print(f"  {drug_name}: {len(ses)}")

# %% [markdown]
# ## 7. Important Caveats
#
# - **Source:** Side effect data comes from **SIDER** (Side Effect
#   Resource), which is derived from FDA drug labels and post-marketing
#   reports. It captures *known, documented* adverse effects.
#
# - **Not exhaustive:** Long-term effects that emerge decades after
#   treatment (e.g., secondary cancers, late cardiac toxicity) may not
#   appear in SIDER if they are studied in survivorship literature
#   rather than drug labels.
#
# - **No severity or frequency:** SPOKE stores the side effect
#   association but not how common or severe each effect is. "Coma"
#   and "nausea" appear with equal weight in the graph.
#
# - **Combination effects:** Side effects listed here are for each drug
#   individually. Combination chemotherapy (vincristine + dactinomycin)
#   may have additive or synergistic toxicity not captured by either
#   drug alone.
#
# - **Drug identifier mapping:** Some drugs have multiple compound
#   entries in SPOKE (different salt forms, stereoisomers), and side
#   effect data may only be attached to ONE of them. Dactinomycin, for
#   example, has three entries in SPOKE — but only the "Dactinomycin
#   (USP)" variant carries SIDER side effect data. The other two have
#   disease associations but zero side effects. This notebook checks
#   all variants and merges their data, but a naive query using just the
#   top search result would return 0 side effects for dactinomycin.
#   This is a real KG usability problem: **the same drug, different
#   identifiers, fragmented data.**
