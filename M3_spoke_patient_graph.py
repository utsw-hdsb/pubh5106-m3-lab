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
# # Querying SPOKE: A Personal Cancer History
#
# **PUBH 5106 — AI in Health Applications, Module 3 Supplement**
#
# This notebook uses the SPOKE API to explore connections between three
# cancers that occurred in a single patient, and asks: **what can a
# knowledge graph tell us that clinical experience did not?**
#
# ## The Patient History
#
# - **Age 7:** Nephroblastoma (Wilms tumor). Treated with vincristine
#   and dactinomycin. Nephrectomy.
# - **Adult:** Malignant carcinoid of the appendix. Partial colectomy.
# - **Adult:** Melanoma (x2). Surgical excision.
# - **Family history:**
#   - Maternal aunt: died of carcinoid syndrome
#   - Maternal grandfather: multiple myeloma (~age 65)
#   - Father: prostate cancer (~age 62)
#   - Sister: breast cancer (age 54)
#
# No clinician has ever connected these diagnoses. Three different cancer
# types in the patient, four more in close relatives, spanning three
# generations. Are they related?

# %%
import json
from collections import defaultdict

import requests
import networkx as nx
from pyvis.network import Network
from IPython.display import display, HTML, Markdown

SPOKE_API = "https://spoke.rbvi.ucsf.edu/api/v1"

# %% [markdown]
# ## 1. Query SPOKE for Each Disease

# %%
def spoke_search(query, node_type="Compound"):
    """Search SPOKE for a node by name."""
    r = requests.get(f"{SPOKE_API}/search/{node_type}/{query}", timeout=30)
    r.raise_for_status()
    return r.json()


def spoke_neighborhood(node_type, identifier):
    """Get 1-hop neighborhood from SPOKE."""
    url = f"{SPOKE_API}/neighborhood/{node_type}/identifier/{identifier}"
    r = requests.post(url, json={"cutoff_Specificity_Index": 0}, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_by_type(data):
    """Parse neighborhood into {node_type: [{name, identifier, neo4j_id}]}."""
    by_type = defaultdict(list)
    for item in data:
        d = item["data"]
        props = d.get("properties", {})
        by_type[d["neo4j_type"]].append({
            "name": props.get("name", "?"),
            "identifier": props.get("identifier", "?"),
            "neo4j_id": d.get("id"),
        })
    return dict(by_type)

# %%
# The three cancers
CANCERS = {
    "Nephroblastoma": "DOID:2154",
    "Melanoma": "DOID:1909",
    # Appendix carcinoid (DOID:0050911) has almost no data in SPOKE.
    # We use the parent term "neuroendocrine tumor" for gene associations.
    "Neuroendocrine tumor\n(parent of carcinoid)": "DOID:169",
}

# The two chemotherapy drugs
DRUGS = {
    "Vincristine": "inchikey:OGWKCGZFUXNPDA-XQKSVPLYSA-N",
    "Dactinomycin": "inchikey:RJURFGZVJUQBHK-IIXSONLDSA-N",
}

# Query all neighborhoods
cancer_data = {}
cancer_genes = {}
cancer_compounds = {}

for name, doid in CANCERS.items():
    data = spoke_neighborhood("Disease", doid)
    parsed = extract_by_type(data)
    cancer_data[name] = parsed
    cancer_genes[name] = {g["name"] for g in parsed.get("Gene", [])}
    cancer_compounds[name] = {c["name"] for c in parsed.get("Compound", [])}
    print(f"{name}: {len(data)} items, "
          f"{len(cancer_genes[name])} genes, "
          f"{len(cancer_compounds[name])} compounds")

# %% [markdown]
# ### The compound identifier problem
#
# Many drugs exist as multiple entries in SPOKE — different salt forms,
# stereoisomers, or naming variants. Side effect data (from SIDER) may
# only be attached to ONE variant. We check all entries and merge.

# %%
# Query drug neighborhoods — check ALL variants for each drug
drug_data = {}
drug_diseases = {}
drug_side_effects = {}

for drug_name in DRUGS:
    # Search for all variants
    variants = spoke_search(drug_name.lower())
    all_diseases = set()
    all_side_effects = set()
    all_data = []

    for v in variants:
        data = spoke_neighborhood("Compound", v["identifier"])
        parsed = extract_by_type(data)
        diseases = {d["name"] for d in parsed.get("Disease", [])}
        ses = {s["name"] for s in parsed.get("SideEffect", [])}
        all_diseases.update(diseases)
        all_side_effects.update(ses)
        all_data.extend(data)
        if ses:
            print(f"  {v['name']} ({v['identifier']}): "
                  f"{len(ses)} side effects")

    drug_data[drug_name] = all_data
    drug_diseases[drug_name] = all_diseases
    drug_side_effects[drug_name] = all_side_effects
    print(f"{drug_name}: {len(all_diseases)} diseases, "
          f"{len(all_side_effects)} side effects (merged from "
          f"{len(variants)} variants)\n")

# %% [markdown]
# ## 2. Find Shared Genes

# %%
cancer_names = list(CANCERS.keys())

print("=== Gene Overlaps ===\n")

# All pairwise
from itertools import combinations
for a, b in combinations(cancer_names, 2):
    shared = cancer_genes[a] & cancer_genes[b]
    a_short = a.split("\n")[0]
    b_short = b.split("\n")[0]
    print(f"{a_short} ∩ {b_short}: {len(shared)} genes")
    if shared:
        for g in sorted(shared):
            print(f"  {g}")
    print()

# All three
shared_all = cancer_genes[cancer_names[0]]
for name in cancer_names[1:]:
    shared_all = shared_all & cancer_genes[name]
print(f"All three: {len(shared_all)} genes")
for g in sorted(shared_all):
    print(f"  {g}")

# %% [markdown]
# ## 3. Check Drug→Cancer Links
#
# Did vincristine or dactinomycin treat any of the other cancers?
# Are any of the patient's later cancers in the drugs' side effect lists?

# %%
print("=== Drug → Disease Connections ===\n")
for drug_name in DRUGS:
    print(f"{drug_name}:")
    for cancer_name in cancer_names:
        # Check if cancer appears in drug's disease list
        cancer_short = cancer_name.split("\n")[0].lower()
        found = [d for d in drug_diseases[drug_name]
                 if cancer_short in d.lower()
                 or ("melanoma" in d.lower() and "melanoma" in cancer_short.lower())
                 or ("carcinoid" in d.lower() and "neuroendocrine" in cancer_short.lower())]
        if found:
            print(f"  → {cancer_name.split(chr(10))[0]}: YES ({', '.join(found[:3])})")
        else:
            print(f"  → {cancer_name.split(chr(10))[0]}: no direct connection")
    print()

print("=== Cancer-Related Side Effects ===\n")
for drug_name in DRUGS:
    ses = drug_side_effects[drug_name]
    cancer_ses = [se for se in ses if any(kw in se.lower()
                  for kw in ["cancer", "neoplasm", "tumor", "malignan",
                             "leuk", "carcin", "melanoma"])]
    if cancer_ses:
        print(f"{drug_name}: {', '.join(cancer_ses)}")
    else:
        print(f"{drug_name}: no cancer-related side effects in SPOKE")
    if ses:
        print(f"  ({len(ses)} total side effects)")
    else:
        print(f"  (no side effect data in SPOKE)")

# %% [markdown]
# ## 4. Build the Patient Graph
#
# Now let's build a focused subgraph showing only the connections relevant
# to this patient's history.

# %%
G = nx.DiGraph()

# -- Colors --
COLOR = {
    "disease": "#e74c3c",       # red
    "gene": "#3498db",          # blue
    "compound": "#f39c12",      # orange
    "side_effect": "#9b59b6",   # purple
    "family": "#1abc9c",        # teal
}

# -- Add the three cancer nodes --
for name in cancer_names:
    short = name.split("\n")[0]
    G.add_node(short, node_type="disease", color=COLOR["disease"],
               label=short, size=30)

# -- Add treatment drugs --
for drug in DRUGS:
    G.add_node(drug, node_type="compound", color=COLOR["compound"],
               label=drug, size=25)

# -- Add treatment edges --
G.add_edge("Vincristine", "Nephroblastoma", relation="TREATS", color="#f39c12")
G.add_edge("Dactinomycin", "Nephroblastoma", relation="TREATS", color="#f39c12")

# -- Add drug→melanoma clinical trial edges --
if "melanoma" in [d.lower() for d in drug_diseases.get("Vincristine", set())]:
    G.add_edge("Vincristine", "Melanoma",
               relation="IN_CLINICAL_TRIALS_FOR", color="#f39c12",
               style="dashed")
if "melanoma" in [d.lower() for d in drug_diseases.get("Dactinomycin", set())]:
    G.add_edge("Dactinomycin", "Melanoma",
               relation="IN_CLINICAL_TRIALS_FOR", color="#f39c12",
               style="dashed")

# -- Add vincristine side effect: Leukaemia --
if "Leukaemia" in drug_side_effects.get("Vincristine", set()):
    G.add_node("Leukaemia\n(side effect)", node_type="side_effect",
               color=COLOR["side_effect"], label="Leukaemia\n(side effect)", size=15)
    G.add_edge("Vincristine", "Leukaemia\n(side effect)",
               relation="CAUSES_SIDE_EFFECT", color="#9b59b6")

# -- Add shared genes --
# Nephroblastoma ↔ Melanoma
nm_genes = cancer_genes[cancer_names[0]] & cancer_genes[cancer_names[1]]
# Nephroblastoma ↔ NET
nn_genes = cancer_genes[cancer_names[0]] & cancer_genes[cancer_names[2]]
# Melanoma ↔ NET
mn_genes = cancer_genes[cancer_names[1]] & cancer_genes[cancer_names[2]]

# Key gene annotations
GENE_ROLES = {
    "TP53": "tumor suppressor\n(Li-Fraumeni)",
    "BRCA2": "DNA repair\n(hereditary cancer)",
    "PALB2": "DNA repair\n(Fanconi pathway)",
    "CTNNB1": "Wnt signaling",
    "HRAS": "RAS oncogene",
    "PIK3CA": "PI3K pathway",
    "NF1": "tumor suppressor\n(neurofibromatosis)",
    "SDHB": "mitochondrial\n(paraganglioma)",
    "SDHC": "mitochondrial\n(paraganglioma)",
    "SDHD": "mitochondrial\n(paraganglioma)",
    "CCND1": "cell cycle\n(cyclin D1)",
}

# Collect all shared genes
all_shared = nm_genes | nn_genes | mn_genes
for gene in sorted(all_shared):
    role = GENE_ROLES.get(gene, "")
    label = f"{gene}\n{role}" if role else gene
    G.add_node(gene, node_type="gene", color=COLOR["gene"],
               label=label, size=20)

    # Add edges to relevant cancers
    for i, name in enumerate(cancer_names):
        short = name.split("\n")[0]
        if gene in cancer_genes[name]:
            G.add_edge(gene, short, relation="ASSOCIATES", color="#3498db")

# -- Add family history --
FAMILY = {
    "Maternal aunt:\ncarcinoid syndrome": "DOID:169",      # → NET
    "Mat. grandfather:\nmultiple myeloma": "DOID:9538",
    "Father:\nprostate cancer": "DOID:10283",
    "Sister:\nbreast cancer (age 54)": "DOID:1612",
}

# Query family cancer gene sets
family_genes = {}
for label, doid in FAMILY.items():
    data = spoke_neighborhood("Disease", doid)
    parsed = extract_by_type(data)
    family_genes[label] = {g["name"] for g in parsed.get("Gene", [])}

for label in FAMILY:
    G.add_node(label, node_type="family", color=COLOR["family"],
               label=label, size=20)

# Connect family members to the most relevant patient cancer
net_short = cancer_names[2].split("\n")[0]
G.add_edge("Maternal aunt:\ncarcinoid syndrome", net_short,
           relation="FAMILY_HISTORY", color="#1abc9c")

# Connect family cancers to shared genes already in the graph
for fam_label, fam_genes_set in family_genes.items():
    for gene in all_shared:
        if gene in fam_genes_set and gene in G.nodes:
            G.add_edge(gene, fam_label,
                       relation="ASSOCIATES", color="#1abc9c")

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# %% [markdown]
# ## 5. Visualize

# %%
net = Network(height="700px", width="100%", directed=True,
              notebook=True, cdn_resources="in_line")

for node, data in G.nodes(data=True):
    net.add_node(node,
                 label=data.get("label", node),
                 color=data.get("color", "#ccc"),
                 size=data.get("size", 15),
                 font={"size": 11, "multi": True},
                 shape="dot")

for src, dst, data in G.edges(data=True):
    dashes = data.get("style") == "dashed"
    net.add_edge(src, dst,
                 label=data.get("relation", ""),
                 arrows="to",
                 color={"color": data.get("color", "#888")},
                 font={"size": 8, "align": "middle"},
                 dashes=dashes)

net.set_options(json.dumps({
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -80,
            "springLength": 150,
            "springConstant": 0.04,
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
    },
    "edges": {"smooth": {"type": "curvedCW", "roundness": 0.15}}
}))

net.save_graph("patient_cancer_graph.html")
with open("patient_cancer_graph.html", encoding="utf-8") as f:
    display(HTML(f.read()))

# %% [markdown]
# ## 5b. Genes Across Patient and Family

# %%
# Which of the shared genes also appear in family members' cancers?
from collections import Counter

all_cancer_genes = {**cancer_genes}
for label in FAMILY:
    all_cancer_genes[label.split("\n")[0]] = family_genes[label]

gene_counts = Counter()
for name, genes in all_cancer_genes.items():
    for g in genes:
        gene_counts[g] += 1

print("=== Genes appearing in 4+ of 7 cancer types "
      "(3 patient + 4 family) ===\n")
print(f"{'Gene':<10} {'Count':>5}  Present in")
print("-" * 70)
for gene, count in gene_counts.most_common():
    if count >= 4:
        present = [n for n, gs in all_cancer_genes.items() if gene in gs]
        # Mark patient cancers with *
        tagged = []
        patient_names = {"Nephroblastoma", "Melanoma",
                         "Neuroendocrine tumor"}
        for p in present:
            short = p.split("\n")[0]
            if short in patient_names:
                tagged.append(f"*{short}*")
            else:
                tagged.append(short)
        print(f"  {gene:<8} {count:>5}  {', '.join(tagged)}")

# %% [markdown]
# **Key findings:**
#
# - **BRCA2** and **PALB2** appear in 4/7 cancers: the patient's
#   nephroblastoma and melanoma, plus the father's prostate cancer and
#   sister's breast cancer. This is the classic BRCA2 family pattern —
#   breast, prostate, and multiple other cancers across generations.
# - **TP53** appears in 5/7 cancers (all except NET and carcinoid).
#   Li-Fraumeni syndrome typically presents with childhood sarcoma or
#   Wilms tumor followed by multiple adult primaries.
# - The **sister's breast cancer at age 54** is below the threshold
#   that triggers BRCA testing in most guidelines. Combined with the
#   patient's three primary cancers and father's prostate cancer,
#   this family would meet criteria for germline panel testing.

# %% [markdown]
# ## 6. What the Graph Reveals
#
# ### Four Categories of Connection
#
# **Drug links (orange):**
# - Vincristine and dactinomycin **treat** nephroblastoma (established)
# - Both are also **in clinical trials for** melanoma
# - Vincristine has **leukaemia** as a known side effect — proof it CAN
#   cause secondary malignancies — but melanoma is NOT in its side
#   effect list
# - Dactinomycin has 75 side effects in SPOKE — but only when you query
#   the right compound variant ("Dactinomycin (USP)"). Two other
#   dactinomycin entries have zero side effects. This fragmentation is a
#   real KG usability problem (see the Side Effect Explorer notebook).
#
# **Gene links (blue → patient cancers):**
# - **HRAS** is the only gene shared across all three patient cancer types
# - **TP53, BRCA2, PALB2, CTNNB1, PIK3CA** connect nephroblastoma to
#   melanoma
# - **NF1, SDHB/C/D** connect melanoma to neuroendocrine tumors — the SDH
#   genes are particularly interesting because germline SDH mutations cause
#   hereditary paraganglioma-pheochromocytoma syndrome AND increase
#   melanoma risk
#
# **Gene links (teal → family cancers):**
# - **BRCA2, PALB2** bridge from the patient's cancers to the sister's
#   breast cancer and father's prostate cancer
# - **TP53** connects to the same family cancers — a second, independent
#   genetic pathway linking the same diseases
# - The maternal grandfather's multiple myeloma shares **TP53** with the
#   patient's cancers
#
# **Family history (teal, direct):**
# - Maternal aunt with carcinoid syndrome → neuroendocrine tumors
# - This is the only family link to the carcinoid/NET branch and suggests
#   a maternal lineage predisposition
#
# ### Hypotheses Strengthened by Family History
#
# 1. **BRCA2/PALB2 hereditary cancer syndrome** (STRONGEST)
#    - Patient: nephroblastoma + melanoma (x2)
#    - Sister: breast cancer at 54
#    - Father: prostate cancer at 62
#    - BRCA2 connects all four. This is a textbook referral pattern for
#      germline testing.
#
# 2. **Li-Fraumeni spectrum (TP53)**
#    - Patient: childhood Wilms tumor + multiple adult primaries
#    - TP53 appears in 5/7 cancers in the family
#    - But Li-Fraumeni typically presents with more aggressive cancers
#      at younger ages than seen here
#
# 3. **SDH/paraganglioma pathway (SDHB/C/D)**
#    - Links melanoma ↔ neuroendocrine tumors specifically
#    - Maternal aunt's carcinoid syndrome fits this pathway
#    - These genes also appear in the sister's breast cancer
#
# 4. **Two lineages, two pathways?**
#    - *Maternal*: aunt with carcinoid → SDH pathway → NET + melanoma
#    - *Paternal*: father with prostate → BRCA2 pathway → breast + melanoma
#    - The patient may carry variants from both sides
#
# ### What a Genetic Counselor Might Recommend
#
# - **Germline panel testing** for the patient: TP53, BRCA2, PALB2,
#   SDHB/C/D at minimum
# - **Cascade testing** for the sister (BRCA2 — actionable for breast
#   cancer surveillance)
# - **Enhanced surveillance** for the patient given three primary cancers
# - **Family pedigree** with cancer ages to formalize the referral
#
# None of this was suggested by any of the patient's treating physicians —
# each cancer was managed in isolation by a different specialty. The
# knowledge graph sees the family as a connected system.

# %% [markdown]
# ## 7. When Your Disease Isn't in the Graph
#
# Appendix carcinoid tumor (`DOID:0050911`) has exactly **3 items** in
# SPOKE — the node itself, its parent "appendix cancer," and one ISA edge.
# **Zero genes. Zero compounds. Zero symptoms.**
#
# This is not because appendix carcinoid is biologically uninteresting.
# It is because:
#
# 1. **Ontology placement:** In the Disease Ontology, appendix carcinoid
#    is classified under "appendix cancer" — NOT under "neuroendocrine
#    tumor." These are separate branches. SPOKE inherits this structure.
#
# 2. **Annotation sparsity:** Gene-disease associations in SPOKE come from
#    curated databases (DisGeNET, OMIM, etc.). Rare diseases at specific
#    anatomical sites often have no curated gene associations — the
#    annotations exist at the parent level ("neuroendocrine tumor") but
#    not the child.
#
# 3. **The proxy problem:** We used "neuroendocrine tumor" (`DOID:169`,
#    47 genes) as a proxy for appendix carcinoid. This is like using
#    "kidney cancer" as a proxy for Wilms tumor — it captures some
#    relevant biology but also includes much that is irrelevant.
#
# This is a fundamental limitation of knowledge graphs: **the level of
# ontological specificity determines what you can find.** A query at
# the wrong level returns either nothing (too specific) or noise
# (too general). Choosing the right level requires domain knowledge
# that the graph itself cannot provide.

# %% [markdown]
# ## 8. Why This Happens: The Ontology Behind the Graph
#
# SPOKE uses the **Disease Ontology (DO)** to organize its disease nodes.
# But DO is just one of several biomedical ontologies. To understand why
# appendix carcinoid is invisible in SPOKE, compare how three major
# ontologies represent the same disease:
#
# | | Disease Ontology (DO) | SNOMED-CT | ICD-10-CM |
# |---|---|---|---|
# | **ID** | [DOID:0050911](http://purl.obolibrary.org/obo/DOID_0050911) | [192681000119104](http://purl.bioontology.org/ontology/SNOMEDCT/192681000119104) | [C7A.020](http://purl.bioontology.org/ontology/ICD10CM/C7A.020) |
# | **Primary axis** | Anatomical site | Compositional (morphology + site + axis) | Body system (for billing) |
# | **Parent class** | appendix cancer | Carcinoid tumor of appendix → Neuroendocrine neoplasm | Malignant carcinoid tumors → Neoplasms |
# | **Linked to NETs?** | No (separate branch) | Yes (via morphology axis) | Partially (same C7A chapter) |
# | **Richness** | ~11,000 terms | ~350,000 concepts | ~72,000 codes |
# | **Design purpose** | Cross-reference hub for research | Clinical documentation + reasoning | Billing + epidemiology |
# | **Access** | Open source | Licensed (UMLS) | Free |
#
# ### What Each Ontology "Sees"
#
# **DO** classifies appendix carcinoid by *where* it is (appendix → GI →
# cancer). The neuroendocrine biology (*what* it is) lives in a separate
# branch. Gene annotations cluster at the parent "neuroendocrine tumor"
# node, which appendix carcinoid is not connected to.
#
# **SNOMED-CT** is *compositional* — it defines concepts through multiple
# axes simultaneously. "Malignant carcinoid tumor of appendix" inherits
# properties from both its morphology (carcinoid/neuroendocrine) AND its
# site (appendix). A SPOKE built on SNOMED could potentially connect
# appendix carcinoid to neuroendocrine gene associations through the
# morphology axis — exactly the link that DO misses.
#
# **ICD-10-CM** groups all carcinoid tumors under C7A regardless of site,
# so appendix carcinoid (C7A.020) sits near small intestine carcinoid
# (C7A.011). But ICD-10 has no gene or biological process associations —
# it exists for counting and billing, not for biological reasoning.
#
# ### Implications for SPOKE
#
# SPOKE chose DO because it is:
# - **Open** (no license required for a public research tool)
# - **A cross-reference hub** (each DO term maps to MeSH, OMIM, ICD,
#   and SNOMED — ideal for integrating 40+ source databases)
# - **Manageable** (11K terms vs. SNOMED's 350K keeps the graph navigable)
#
# The cost: DO's single-axis classification (primarily by site) means
# diseases that share *biology* but not *location* may not be connected.
# Appendix carcinoid and pancreatic neuroendocrine tumor share
# neuroendocrine biology, but in DO they live in different branches
# (appendix cancer vs. pancreatic cancer). SPOKE inherits this blind spot.
#
# A SPOKE built on SNOMED's multi-axis structure might have surfaced the
# neuroendocrine connection directly. But it would also be 30x larger,
# require licensing, and be far harder to traverse. This is a real
# engineering trade-off, not a design error — and it is the kind of
# **ontological commitment** (Davis et al., 1993) that shapes what a
# knowledge system can and cannot reason about.

# %% [markdown]
# ## 9. Lessons for Knowledge Graph Design
#
# This case demonstrates several KG principles from the M3 lab:
#
# | Principle | How It Appears Here |
# |-----------|-------------------|
# | **Entity grounding matters** | "Appendix carcinoid" has 0 genes in SPOKE; "neuroendocrine tumor" has 47. The ontology level you choose determines what you find. |
# | **Edge types encode meaning** | TREATS vs. IN_CLINICAL_TRIALS_FOR vs. CAUSES_SIDE_EFFECT are three very different relationships connecting a drug to a disease. Collapsing them would be misleading. |
# | **Absence ≠ evidence of absence** | Melanoma is not in either drug's side effect list. Dactinomycin appeared to have NO side effects at all until we checked alternate compound identifiers. Both kinds of absence — missing associations and fragmented identifiers — can mislead. |
# | **KGs generate hypotheses** | SPOKE cannot tell us which hypothesis is correct. It surfaces the *structure* of the question — shared genes, shared drugs, family history — and a human decides what to test. |
# | **Silos hide connections** | Three different specialists treated three different cancers. The knowledge graph sees all three simultaneously and finds the genetic thread connecting them. |
# | **Family context changes everything** | The patient's cancers alone suggest several hypotheses. Adding four family cancers narrows the field dramatically — BRCA2/PALB2 becomes the leading hypothesis because the sister and father's cancers fit the same pattern. |
# | **Rare diseases are invisible** | Appendix carcinoid has no gene data in SPOKE. The graph cannot help with what it doesn't know. |
# | **Ontology choice is an ontological commitment** | SPOKE uses DO (site-based), not SNOMED (multi-axis). This determines which diseases are connected and which are invisible. A different ontology choice would yield a different graph with different findings. |
# | **Identifier fragmentation hides data** | Dactinomycin has 3 compound entries in SPOKE. Side effects are attached to only one (the USP variant). A naive query returns 0 side effects for a major chemotherapy drug. The same drug, different identifiers, fragmented knowledge. |

# %%
