# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploring SPOKE: From Wilms Tumor to Melanoma
#
# **PUBH 5106 — AI in Health Applications, Module 3 Supplement**
#
# This notebook demonstrates how to query the SPOKE biomedical knowledge
# graph programmatically using its public REST API. We use a real clinical
# question as our guide: **is there a connection between childhood Wilms
# tumor (nephroblastoma) and adult melanoma?**
#
# ## What is SPOKE?
#
# SPOKE (Scalable Precision Medicine Open Knowledge Engine) is a biomedical
# knowledge graph built by the Baranzini Lab at UCSF. It integrates 40+
# databases into a single graph with ~27 million nodes and ~53 million edges.
# Node types include diseases, genes, compounds, proteins, symptoms, anatomy,
# and more. Edge types encode relationships like `TREATS`, `ASSOCIATES`,
# `CAUSES`, and `ISA`.
#
# **API documentation:** https://spoke.rbvi.ucsf.edu/swagger/
#
# ## The Clinical Question
#
# A patient had a Wilms tumor (nephroblastoma) at age 7, treated with
# chemotherapy including vincristine. As an adult, the same patient
# developed melanoma twice. No clinician has ever connected these diagnoses.
#
# Using the SPOKE Neighborhood Explorer, we noticed that nephroblastoma
# and melanoma are both connected to **vincristine** — but through different
# edge types (treatment vs. clinical trial). This raises several hypotheses:
#
# 1. Could vincristine (or other chemotherapy) increase melanoma risk?
# 2. Are there shared genetic pathways between these cancers?
# 3. Is the connection through shared drugs, shared biology, or both?
#
# Let's use the SPOKE API to find out.

# %%
import json
from collections import defaultdict

import requests
import pandas as pd
from IPython.display import display, Markdown

SPOKE_API = "https://spoke.rbvi.ucsf.edu/api/v1"

# %% [markdown]
# ## 1. Finding Our Diseases in SPOKE
#
# First, let's search for nephroblastoma and melanoma to get their
# SPOKE identifiers.

# %%
def spoke_search(query, node_type=None):
    """Search SPOKE for a term. Optionally filter by node type."""
    if node_type:
        url = f"{SPOKE_API}/search/{node_type}/{query}"
    else:
        url = f"{SPOKE_API}/search/{query}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def spoke_neighborhood(node_type, identifier):
    """Get the 1-hop neighborhood of a node."""
    url = f"{SPOKE_API}/neighborhood/{node_type}/identifier/{identifier}"
    r = requests.post(url, json={"cutoff_Specificity_Index": 0}, timeout=60)
    r.raise_for_status()
    return r.json()


def parse_neighborhood(data):
    """Parse a neighborhood response into categorized nodes and edge types."""
    nodes_by_type = defaultdict(list)
    edge_types = defaultdict(int)
    for item in data:
        d = item["data"]
        t = d["neo4j_type"]
        props = d.get("properties", {})
        if "_" in t and t == t.upper():
            # Edge type entry
            edge_types[t] += 1
        else:
            nodes_by_type[t].append({
                "name": props.get("name", "?"),
                "identifier": props.get("identifier", "?"),
                "neo4j_id": d.get("id"),
            })
    return dict(nodes_by_type), dict(edge_types)

# %%
# Search for our diseases
for disease in ["nephroblastoma", "melanoma"]:
    results = spoke_search(disease, "Disease")
    print(f"\n=== {disease} ===")
    for r in results[:3]:
        print(f"  {r['name']} ({r['identifier']}) — score: {r['score']:.1f}")

# %% [markdown]
# **Nephroblastoma** is `DOID:2154` and **melanoma** is `DOID:1909`.
#
# Note: "Renal Wilms' tumor" (`DOID:5176`) is a child of nephroblastoma in the
# Disease Ontology hierarchy, connected by an `ISA_DiD` edge. We'll use
# the parent term since it has more connections.

# %% [markdown]
# ## 2. Exploring Nephroblastoma's Neighborhood
#
# What does SPOKE know about nephroblastoma?

# %%
nephro_data = spoke_neighborhood("Disease", "DOID:2154")
nephro_nodes, nephro_edges = parse_neighborhood(nephro_data)

print(f"Nephroblastoma: {len(nephro_data)} total items\n")
print("Node types:")
for t, items in sorted(nephro_nodes.items()):
    print(f"  {t}: {len(items)}")
print("\nEdge types:")
for t, count in sorted(nephro_edges.items()):
    print(f"  {t}: {count}")

# %%
# What compounds treat nephroblastoma?
print("Compounds connected to nephroblastoma:")
for c in sorted(nephro_nodes.get("Compound", []), key=lambda x: x["name"]):
    print(f"  {c['name']}")

# %%
# What genes are associated?
print("Genes associated with nephroblastoma:")
for g in sorted(nephro_nodes.get("Gene", []), key=lambda x: x["name"]):
    print(f"  {g['name']}")

# %% [markdown]
# ## 3. The Vincristine Connection
#
# Vincristine appears in nephroblastoma's compound neighborhood. Let's explore
# vincristine itself — what diseases does it connect to, and how?

# %%
# Find vincristine's identifier
vincristine_id = None
for c in nephro_nodes.get("Compound", []):
    if c["name"] == "Vincristine":
        vincristine_id = c["identifier"]
        break
print(f"Vincristine identifier: {vincristine_id}")

# %%
vinc_data = spoke_neighborhood("Compound", vincristine_id)
vinc_nodes, vinc_edges = parse_neighborhood(vinc_data)

print(f"Vincristine: {len(vinc_data)} total items\n")
print("Node types:")
for t, items in sorted(vinc_nodes.items()):
    print(f"  {t}: {len(items)}")
print("\nEdge types:")
for t, count in sorted(vinc_edges.items()):
    print(f"  {t}: {count}")

# %%
# Is melanoma in vincristine's disease connections?
vinc_diseases = {d["name"].lower(): d for d in vinc_nodes.get("Disease", [])}
if "melanoma" in vinc_diseases:
    print("Melanoma IS connected to vincristine")
    print(f"  {vinc_diseases['melanoma']}")
else:
    print("Melanoma is NOT directly connected to vincristine")

# %% [markdown]
# ### Side Effects: Does SPOKE Know About Second Cancers?
#
# SPOKE includes side effect data from SIDER (Side Effect Resource).
# The edge type `CAUSES_CcSE` connects compounds to known side effects.
# Let's check if any of vincristine's side effects are cancer-related.

# %%
side_effects = sorted([se["name"] for se in vinc_nodes.get("SideEffect", [])])
print(f"Vincristine has {len(side_effects)} known side effects in SPOKE\n")

# Check for cancer-related side effects
cancer_ses = [se for se in side_effects
              if any(kw in se.lower() for kw in
                     ["cancer", "neoplasm", "tumor", "tumour",
                      "malignan", "melanoma", "carcino", "leuk"])]
if cancer_ses:
    print("Cancer-related side effects:")
    for se in cancer_ses:
        print(f"  {se}")
else:
    print("No cancer-related side effects listed (except possibly leukemia)")
    # Check specifically
    leuk = [se for se in side_effects if "leuk" in se.lower() or "leukaemia" in se.lower()]
    if leuk:
        print(f"\n  Exception: {leuk}")

print(f"\nAll side effects:")
for se in side_effects:
    print(f"  {se}")

# %% [markdown]
# **Key finding:** Vincristine's SIDER-derived side effects include
# **Leukaemia** (a secondary malignancy) but NOT melanoma or any solid
# tumor. This means:
#
# - The adverse-effect databases do not support a vincristine→melanoma link
# - But absence from SIDER is not evidence of absence — SIDER captures
#   known, reported side effects, not long-term epidemiological associations
# - Secondary malignancies after childhood cancer treatment are studied in
#   survivorship literature, not typically in drug side effect databases

# %% [markdown]
# ## 4. The Bigger Question: Shared Biology
#
# The vincristine link is through a shared drug. But are there
# **biological** connections between nephroblastoma and melanoma?
# Shared genes would suggest a common predisposition rather than
# a treatment effect.

# %%
# Get melanoma's neighborhood
mel_data = spoke_neighborhood("Disease", "DOID:1909")
mel_nodes, mel_edges = parse_neighborhood(mel_data)

print(f"Melanoma: {len(mel_data)} total items")
print(f"  Genes: {len(mel_nodes.get('Gene', []))}")
print(f"  Compounds: {len(mel_nodes.get('Compound', []))}")

# %%
# Find shared genes
nephro_genes = {g["name"] for g in nephro_nodes.get("Gene", [])}
mel_genes = {g["name"] for g in mel_nodes.get("Gene", [])}
shared_genes = nephro_genes & mel_genes

print(f"Nephroblastoma genes: {len(nephro_genes)}")
print(f"Melanoma genes: {len(mel_genes)}")
print(f"Shared genes: {len(shared_genes)}")
print(f"\n{'Gene':<12} Known Role")
print("-" * 60)
gene_roles = {
    "TP53": "Tumor suppressor; most frequently mutated gene in cancer",
    "BRCA2": "DNA repair (homologous recombination); hereditary cancer risk",
    "PALB2": "BRCA2 partner; Fanconi anemia pathway; multi-cancer risk",
    "CTNNB1": "Wnt/beta-catenin signaling; driver in Wilms and melanoma",
    "HRAS": "RAS oncogene family; cell growth signaling",
    "PIK3CA": "PI3K pathway; oncogenic driver in many cancer types",
}
for gene in sorted(shared_genes):
    role = gene_roles.get(gene, "")
    print(f"  {gene:<10} {role}")

# %%
# Find shared compounds
nephro_compounds = {c["name"] for c in nephro_nodes.get("Compound", [])}
mel_compounds = {c["name"] for c in mel_nodes.get("Compound", [])}
shared_compounds = nephro_compounds & mel_compounds

print(f"\nShared compounds: {len(shared_compounds)}")
for c in sorted(shared_compounds):
    print(f"  {c}")

# %% [markdown]
# ### What the Shared Genes Tell Us
#
# The 6 shared genes are not random — they fall into specific categories:
#
# | Category | Genes | Implication |
# |----------|-------|-------------|
# | DNA repair | BRCA2, PALB2 | Germline variants predispose to multiple cancer types |
# | Tumor suppression | TP53 | Li-Fraumeni syndrome causes childhood AND adult cancers |
# | Oncogenic signaling | CTNNB1, HRAS, PIK3CA | Shared driver pathways |
#
# **BRCA2 and PALB2** are particularly interesting: they are part of the
# Fanconi anemia DNA repair pathway. Germline mutations in these genes
# are known to increase risk for breast, ovarian, pancreatic, and other
# cancers. A child with a Wilms tumor who also carries a germline BRCA2
# or PALB2 variant might have elevated adult cancer risk — **independent
# of any treatment effect.**
#
# **TP53** is even more directly relevant: germline TP53 mutations cause
# Li-Fraumeni syndrome, which is characterized by childhood sarcomas,
# brain tumors, AND Wilms tumor, with dramatically elevated lifetime
# risk of multiple primary cancers including melanoma.

# %% [markdown]
# ## 5. Generating Hypotheses
#
# SPOKE has helped us move from a single observation (vincristine connects
# these diseases) to three distinct hypotheses:

# %%
display(Markdown("""
### Hypothesis 1: Treatment Effect (Vincristine)
**Path:** nephroblastoma ←[TREATS]— vincristine —[IN_CLINICAL_TRIALS_FOR]→ melanoma

**Evidence for:** Vincristine has "Leukaemia" as a known side effect,
showing it CAN cause secondary malignancies.

**Evidence against:** Melanoma is NOT in vincristine's side effect list (SIDER).
The clinical trial edge means vincristine is being *tested against* melanoma,
not that it causes it.

**Strength:** Weak. The vincristine→melanoma edge is a treatment trial,
not a causal link.

---

### Hypothesis 2: Shared Genetic Predisposition (DNA Repair)
**Path:** nephroblastoma ←[ASSOCIATES]— BRCA2/PALB2 —[ASSOCIATES]→ melanoma

**Evidence for:** BRCA2 and PALB2 are established multi-cancer risk genes.
Germline variants in these genes could explain both childhood Wilms tumor
and adult melanoma in the same patient.

**Evidence against:** BRCA2/PALB2 are associated with MANY cancers. The
association may be too broad to be informative for this specific pair.

**Strength:** Moderate. Would require germline testing to evaluate.

---

### Hypothesis 3: Li-Fraumeni Spectrum (TP53)
**Path:** nephroblastoma ←[ASSOCIATES]— TP53 —[ASSOCIATES]→ melanoma

**Evidence for:** Germline TP53 mutations cause Li-Fraumeni syndrome,
which includes both childhood renal tumors and adult melanoma. This is
the most specific genetic explanation for this particular cancer pair.

**Evidence against:** Li-Fraumeni typically presents with multiple primary
cancers — two melanomas alone might not be sufficient to suspect LFS.

**Strength:** Moderate to strong. Testable with germline TP53 sequencing.
"""))

# %% [markdown]
# ## 6. What SPOKE Can and Cannot Tell Us
#
# | SPOKE can... | SPOKE cannot... |
# |-------------|----------------|
# | Show which entities are connected | Tell you the *strength* of a connection |
# | Reveal shared genes between diseases | Distinguish germline from somatic variants |
# | List known drug side effects (SIDER) | Detect long-term epidemiological associations |
# | Surface multiple hypotheses quickly | Rank which hypothesis is most likely |
# | Point you to the right databases | Replace clinical judgment or genetic testing |
#
# The key insight: **SPOKE is a hypothesis generation tool, not a
# hypothesis testing tool.** It surfaces connections that a human
# can then investigate through literature review, clinical databases,
# or genetic testing.
#
# In this case, SPOKE moved us from "vincristine connects these diseases"
# (a drug-centered hypothesis) to "shared DNA repair genes connect these
# diseases" (a biology-centered hypothesis) in minutes. A literature
# search would likely confirm that childhood cancer survivors have
# elevated second primary cancer risk through both treatment effects
# AND germline predisposition — but SPOKE helped us see the structure
# of the question before reading a single paper.

# %% [markdown]
# ## 7. Connection to the Lab
#
# In the M3 lab, you built a knowledge graph from 20 sentences about CKD.
# Your graph had ~30 nodes and ~25 edges. SPOKE has 27 million nodes and
# 53 million edges. But the underlying logic is the same:
#
# - Both represent knowledge as **subject-predicate-object triples**
# - Both require **entity grounding** (your Gilda step = SPOKE's ontology mapping)
# - Both have **schema constraints** (your allowed predicates = SPOKE's 93 edge types)
# - Both have **gaps** — your graph missed ARBs; SPOKE doesn't capture
#   long-term survivorship associations
#
# The difference is scale and curation. Your graph was built in 2 hours
# by an LLM reading prose. SPOKE was built over years by integrating
# 40+ curated databases. The next frontier — **KG-RAG** (Soman et al.
# 2024) — uses LLMs to *query* graphs like SPOKE in natural language,
# combining the auditability of structured knowledge with the fluency
# of language models.
