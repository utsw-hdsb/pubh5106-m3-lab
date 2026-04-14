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
# **Platform:** GitHub Codespaces

# %% [markdown]
# # M3 Lab: Knowledge Engineer
#
# **PUBH 5106 — AI in Health Applications**
#
# ## Learning Objectives
#
# By the end of this lab, you will be able to:
#
# 1. Describe how a knowledge graph represents biomedical facts as
#    subject-predicate-object (SPO) triples
# 2. Write an extraction prompt that instructs an LLM to produce structured
#    SPO triples from clinical text
# 3. Ground extracted entities to standard biomedical ontologies using Gilda
# 4. Compare constrained (schema-locked) vs. open extraction and explain
#    the trade-offs
# 5. Compare a general-purpose LLM with domain-specific fine-tuned models
#    (MedSPO-3B and MedSPO-7B) and explain when each approach is appropriate
# 6. Evaluate the robustness of an extraction pipeline on unseen text
#
# ## How This Lab Works
#
# This is a **competitive group lab** with six scored rounds. Each round
# builds on the neurosymbolic pipeline from the pre-class material: an LLM
# extracts knowledge (neural), and symbolic tools ground and structure it.
#
# An automated scoring function evaluates your extracted triples against a
# gold standard, and a shared leaderboard tracks performance in real time.
#
# | Round | Name | Time | What You Do |
# |-------|------|------|-------------|
# | --- | SPOKE Warm-Up | ~10 min | Explore a professional KG (not scored) |
# | 1 | Cold Extract | ~12 min | Bare-minimum prompt, small model |
# | 2 | Knowledge Engineer | ~25 min | Design system prompt + rules |
# | 3 | Ground Truth | ~15 min | Add entity grounding with Gilda |
# | 4 | Schema Lock | ~12 min | Constrained entity types + predicates |
# | 5 | The Specialist | ~10 min | Pre-computed fine-tuned models (MedSPO-3B & 7B) |
# | 6 | New Territory | ~12 min | Extract from a lipid management article |
# | --- | Reflection | ~12 min | Leaderboard + discussion |

# %% [markdown]
# ---
# ## Setup

# %%
import json
from pathlib import Path

import pandas as pd
import networkx as nx
from pyvis.network import Network
from IPython.display import display, Markdown, HTML

import lab_utils as lu

# %% [markdown]
# ### Your Group Name and API Keys
#
# Set your group name and register your team's Groq API keys.
# Each team member should create a free account at
# [console.groq.com](https://console.groq.com) and generate an API key.

# %%
lu.GROUP_NAME = "CHANGE_ME"  # <-- Set your group name here

# %%
lu.set_api_keys([
    "gsk_...",  # Team member 1
    "gsk_...",  # Team member 2
    "gsk_...",  # Team member 3
    # Add more keys if your team has more members
])

# %% [markdown]
# ### Verify Setup

# %%
lu.verify_setup()

# %% [markdown]
# ### The Target Output Format
#
# For each sentence, the LLM should return a JSON array of SPO triples:

# %%
TARGET_FORMAT = """\
[
  {"subject": "hypertension", "predicate": "causes", "object": "chronic kidney disease"},
  {"subject": "ACE inhibitors", "predicate": "treats", "object": "proteinuria"}
]"""

print("Target triple format:")
print(TARGET_FORMAT)

# %% [markdown]
# ---
# ## SPOKE Warm-Up (~10 min, not scored)
#
# Before building anything, explore **SPOKE** -- UCSF's biomedical knowledge
# graph with 27 million nodes and 53 million edges.
#
# **Go to:** [SPOKE Neighborhood Explorer](https://spoke.rbvi.ucsf.edu)
#
# 1. In the toolbar, click **Source** and select **"Disease (identifier)"**
#    from the dropdown
# 2. Type **DOID:784** in the text field (this is chronic kidney disease)
#    and click **Submit**
# 3. The graph is dense -- that's the point. CKD touches genes, proteins,
#    compounds, anatomy, and more. Check the **Legend** panel on the right
#    to see the node types (each color = one type).
# 4. Use **"Find in network"** to search for **lisinopril**. Drag the
#    highlighted node to pull it away from the cluster. What diseases is
#    lisinopril directly connected to?
# 5. Notice what is **missing**: SPOKE connects lisinopril to diseases
#    it treats, but not to "ACE inhibitor" (drug class) or "proteinuria"
#    (symptom). Those relationships exist in clinical text but are not in
#    SPOKE's curated database. You will extract them in the lab.
#
# Record observations below.

# %%
spoke_observations = """
How many node types does SPOKE use for CKD's neighborhood?
(count the colors in the Legend)


Diseases connected to lisinopril (drag the node to see):
1.
2.
3.

What relationship from the CKD article is NOT in SPOKE?
(hint: think about drug classes, symptoms, lab values)


"""
print(spoke_observations)

# %% [markdown]
# ---
# ## Round 1: Cold Extract (~12 min)
#
# **No system prompt. No rules. Just a user message and a small model.**
#
# The default `extract_all_triples()` sends each sentence to the LLM with
# a bare user message: "Extract SPO triples from this sentence." No system
# prompt guides the model's behavior.
#
# **Rules:**
# - No `system_prompt` argument
# - Small model only
# - Temperature 0.0

# %% [markdown]
# ### 1.1 Preview the Article

# %%
print("CKD Article (20 sentences):\n")
for i, s in enumerate(lu.CKD_SENTENCES, 1):
    print(f"  [{i:>2}] {s[:100]}{'...' if len(s) > 100 else ''}")

# %% [markdown]
# ### 1.2 Extract

# %%
r1_triples = lu.extract_all_triples(
    lu.CKD_SENTENCES,
    system_prompt=None,
    model=lu.SMALL_MODEL,
    temperature=0.0,
)

# %%
r1_result = lu.show_triple_score(
    r1_triples, lu.GOLD_TRIPLES["triples"],
    label="Round 1 -- Cold Extract",
)
lu.submit_to_leaderboard(1, r1_result)

# %%
# Inspect what was extracted
df_r1 = pd.DataFrame(r1_triples)
if not df_r1.empty:
    display(df_r1[["subject", "predicate", "object"]].head(20))
else:
    print("No triples extracted.")

# %% [markdown]
# **Reflection:** What went wrong? Common issues:
# - LLM returns prose instead of JSON arrays
# - Vague subjects/objects ("the condition", "patients")
# - Predicates inconsistent or too verbose
# - Missing important relationships

# %% [markdown]
# ---
# ## Round 2: Knowledge Engineer (~25 min)
#
# **Unlocked:** System prompt.
#
# You are now a **knowledge engineer**. Design a system prompt that tells
# the LLM:
# - What it is (role)
# - What SPO triples are
# - What entity types are valid
# - What predicates to use
# - What to exclude (negations? hedged statements? vague references?)
# - What output format to return
#
# This is the same work MYCIN's authors did when they wrote production rules.
# The quality of your rules determines the quality of the extraction.

# %%
# -- YOUR ROUND 2 SYSTEM PROMPT --
# TODO: Write a system prompt that instructs the LLM to extract SPO triples.
# Think about: role definition, entity types, predicate vocabulary,
# exclusion rules, and output format.

r2_system_prompt = """
You are a biomedical knowledge graph builder.
Extract subject-predicate-object (SPO) triples from biomedical text.

ENTITY RULES:
- Subjects and objects must be specific biomedical entities: diseases, drugs,
  drug classes, symptoms, lab values, anatomical structures, or procedures.
- Do NOT use pronouns, generic terms ("patients", "individuals"), or
  vague references as subjects or objects.
- Use the most specific term available (e.g., "ACE inhibitors" not
  "medications").

PREDICATE RULES:
- Use concise, consistent predicates: causes, treats, is_complication_of,
  is_marker_of, is_risk_factor_for, manages, reduces, worsens,
  slows_progression_of, calculated_from, characterized_by, may_slow.
- Extract one triple per distinct relationship. Do not create duplicate
  or redundant triples from the same sentence.

OUTPUT RULES:
- Return ONLY a valid JSON array: [{"subject": "...", "predicate": "...", "object": "..."}]
- No prose, no explanation, no markdown formatting.
- If no valid triple exists, return [].
- Extract at most 3 triples per sentence."""

# %%
r2_triples = lu.extract_all_triples(
    lu.CKD_SENTENCES,
    system_prompt=r2_system_prompt,
    model=lu.SMALL_MODEL,
    temperature=0.0,
)

# %%
r2_result = lu.show_triple_score(
    r2_triples, lu.GOLD_TRIPLES["triples"],
    label="Round 2 -- Knowledge Engineer",
)
lu.submit_to_leaderboard(2, r2_result)

# %%
# Compare Round 1 vs Round 2
print("\n=== Prompt Impact ===")
print(f"  Round 1 (no prompt):   F1={r1_result['triple_f1']}, "
      f"matched={r1_result['matched']}/{len(lu.GOLD_TRIPLES['triples'])}")
print(f"  Round 2 (engineered):  F1={r2_result['triple_f1']}, "
      f"matched={r2_result['matched']}/{len(lu.GOLD_TRIPLES['triples'])}")

# %% [markdown]
# **Discussion:** Which rules in your system prompt had the most impact?
# How does this connect to Davis et al. (1993) -- the idea that knowledge
# representation requires explicit "ontological commitments"?

# %% [markdown]
# ---
# ## Round 3: Ground Truth (~15 min)
#
# **Unlocked:** Entity grounding with Gilda.
#
# Your Round 2 triples use whatever surface forms the LLM chose. "CKD",
# "chronic kidney disease", and "Chronic Kidney Disease" would be three
# different nodes in your graph — even though they refer to the same
# concept. **Entity grounding** solves this by mapping every extracted
# term to a standardized identifier from a biomedical ontology.
#
# **[Gilda](https://github.com/gyorilab/gilda)** is an entity grounding
# tool developed at Harvard Medical School. Given a text string like
# "chronic kidney disease", Gilda searches across multiple biomedical
# ontologies (MeSH, ChEBI, Disease Ontology, HGNC, GO, and others) and
# returns the best-matching concept with its standardized ID. For example:
#
# - "chronic kidney disease" → `MESH:D007676`
# - "ACE inhibitors" → `CHEBI:35457`
# - "eGFR" → `EFO:0005208`
#
# This is the same function that SPOKE performs when it builds its
# knowledge graph — every node in SPOKE is grounded to a standardized
# ontology. The difference: SPOKE achieves 99.7% grounding because its
# entities are curated. Your LLM-extracted entities will ground at a
# much lower rate, and the failures reveal where the LLM's language
# diverges from standard biomedical terminology.
#
# This round adds grounding to your pipeline. Your composite score now
# includes grounding rate (70% triple F1 + 30% grounding rate).
#
# Keep your Round 2 system prompt.

# %% [markdown]
# ### 3.1 Test Gilda
#
# Try grounding some known CKD terms. Notice which succeed, which fail,
# and what ontology each maps to:

# %%
import gilda

test_terms = [
    "chronic kidney disease", "hypertension", "diabetes mellitus",
    "ACE inhibitors", "eGFR", "proteinuria", "hemodialysis",
    "anemia", "NSAIDs", "hyperkalemia",
]
print(f"{'Term':<35} {'Result'}")
print("-" * 75)
for term in test_terms:
    try:
        matches = gilda.ground(term)
        if matches:
            top = matches[0]
            print(f"{term:<35} {top.term.db}:{top.term.id} "
                  f"(score={top.score:.3f})")
        else:
            print(f"{term:<35} -- no match")
    except Exception:
        print(f"{term:<35} -- error")

# %% [markdown]
# ### 3.2 Ground Your Round 2 Triples

# %%
r3_groundings = lu.ground_entities(r2_triples)

matched = sum(1 for v in r3_groundings.values() if v is not None)
total = len(r3_groundings)
print(f"\nGrounded {matched}/{total} entities ({matched/total*100:.0f}%)")

# Show grounding details
print(f"\n{'Entity':<40} {'Status'}")
print("-" * 75)
for entity, result in sorted(r3_groundings.items()):
    if result:
        print(f"{entity:<40} {result['id']} "
              f"(score={result['score']})")
    else:
        print(f"{entity:<40} -- no match")

# %%
r3_result = lu.show_triple_score(
    r2_triples, lu.GOLD_TRIPLES["triples"],
    label="Round 3 -- Ground Truth",
    groundings=r3_groundings,
)
lu.submit_to_leaderboard(3, r3_result)

# %% [markdown]
# ### 3.3 Visualize Your Knowledge Graph

# %%
G = nx.DiGraph()

for t in r2_triples:
    subj = t["subject"].lower().strip()
    obj = t["object"].lower().strip()
    pred = t["predicate"].lower().strip()

    for entity in [subj, obj]:
        if entity not in G.nodes:
            g = r3_groundings.get(entity)
            G.add_node(entity,
                       grounded=g is not None,
                       ontology_id=g["id"] if g else "",
                       gilda_score=g["score"] if g else 0.0)
    G.add_edge(subj, obj, predicate=pred)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
grounded_n = sum(1 for _, d in G.nodes(data=True) if d["grounded"])
print(f"Grounded: {grounded_n}/{G.number_of_nodes()}")

# Top connected nodes
print("\nTop 5 most connected:")
for node, deg in sorted(G.degree(), key=lambda x: -x[1])[:5]:
    mark = "G" if G.nodes[node]["grounded"] else " "
    print(f"  [{mark}] {node}: {deg} connections")

# %%
# Interactive visualization
net = Network(height="600px", width="100%", directed=True,
              notebook=True, cdn_resources="in_line")

for node, data in G.nodes(data=True):
    color = "#4a90d9" if data["grounded"] else "#f5a623"
    tooltip = f"{node}"
    if data["grounded"]:
        tooltip += f"\n{data['ontology_id']} (score={data['gilda_score']})"
    net.add_node(node, label=node, title=tooltip, color=color,
                 size=12 + 4 * G.degree(node), font={"size": 12})

for src, dst, data in G.edges(data=True):
    net.add_edge(src, dst, label=data["predicate"], arrows="to",
                 font={"size": 9, "align": "middle"},
                 color={"color": "#888"})

net.set_options(json.dumps({
    "physics": {
        "forceAtlas2Based": {"gravitationalConstant": -50,
                             "springLength": 120},
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
    },
    "edges": {"smooth": {"type": "curvedCW", "roundness": 0.2}}
}))

net.save_graph("m3_kg_visualization.html")
with open("m3_kg_visualization.html", encoding="utf-8") as f:
    display(HTML(f.read()))

# %% [markdown]
# **Discussion:**
# - Which entities failed to ground? Why?
# - How does your grounding rate compare to SPOKE's 99.7%?
# - What would it take to close the gap?

# %% [markdown]
# ---
# ## Round 4: Schema Lock (~12 min)
#
# **Unlocked:** Constrained entity types and predicate vocabulary.
#
# In Rounds 1-3, the LLM could use any predicate it wanted. Now you will
# **lock the schema** -- specifying exactly which entity types and predicates
# are allowed. This mirrors how SPOKE uses typed edges like `TREATS_CtD`.
#
# **Rules:**
# - Use the constrained system prompt below (or modify it)
# - Same sentences, same small model
# - Score uses grounding (same as Round 3)

# %%
# Allowed entity types and predicates for schema-locked extraction
ALLOWED_ENTITY_TYPES = [
    "Disease", "Drug", "Symptom", "LabValue", "Procedure", "Anatomy",
]

ALLOWED_PREDICATES = [
    "causes", "treats", "is_complication_of", "is_marker_of",
    "is_risk_factor_for", "manages", "reduces", "worsens",
    "slows_progression_of", "calculated_from", "characterized_by",
]

# %%
# -- YOUR ROUND 4 SYSTEM PROMPT --
# TODO: Write a schema-constrained system prompt that uses the allowed
# entity types and predicates above. Build on your Round 2 prompt but add
# explicit constraints.

r4_system_prompt = """
You are a biomedical knowledge graph builder.
Extract subject-predicate-object triples from biomedical text.

ALLOWED ENTITY TYPES (subjects and objects must be one of these):
- Disease (e.g., chronic kidney disease, hypertension, anemia)
- Drug (e.g., ACE inhibitors, NSAIDs, phosphate binders)
- Symptom (e.g., proteinuria, edema, hyperkalemia)
- LabValue (e.g., eGFR, serum creatinine, GFR)
- Procedure (e.g., hemodialysis, kidney transplantation)
- Anatomy (e.g., kidney, nephron)

ALLOWED PREDICATES (use only these):
- causes
- treats
- is_complication_of
- is_marker_of
- is_risk_factor_for
- manages
- reduces
- worsens
- slows_progression_of
- calculated_from
- characterized_by

RULES:
- If a relationship does not fit the allowed predicates, skip it.
- Subjects and objects must be specific named entities, not pronouns or
  generic terms like "patients".
- Return ONLY a valid JSON array: [{"subject": "...", "predicate": "...", "object": "..."}]
- Return [] if no valid triple exists.
"""

# %%
r4_triples = lu.extract_all_triples(
    lu.CKD_SENTENCES,
    system_prompt=r4_system_prompt,
    model=lu.SMALL_MODEL,
    temperature=0.0,
)

# %%
r4_groundings = lu.ground_entities(r4_triples)

r4_result = lu.show_triple_score(
    r4_triples, lu.GOLD_TRIPLES["triples"],
    label="Round 4 -- Schema Lock",
    groundings=r4_groundings,
)
lu.submit_to_leaderboard(4, r4_result)

# %%
# Compare open vs. constrained
print("\n=== Open vs. Constrained Extraction ===")
print(f"  Round 2 (open):        F1={r2_result['triple_f1']}, "
      f"extracted={r2_result['matched']+r2_result['extra']} triples")
print(f"  Round 4 (constrained): F1={r4_result['triple_f1']}, "
      f"extracted={r4_result['matched']+r4_result['extra']} triples")
print(f"\n  Round 3 grounding:     {r3_result['grounding_rate']:.0%}")
print(f"  Round 4 grounding:     {r4_result['grounding_rate']:.0%}")

# %%
from lab_utils import match_triple                        
print("=== Missed gold triples (Round 4) ===")
for j, g in enumerate(lu.GOLD_TRIPLES["triples"]):                                                                                                                                                                 
  matched = any(match_triple(ext, g) for ext in r4_triples)                                                                                                                                                      
  if not matched:                                                                                                                                                                                                
      print(f"  MISS: {g['subject']} --[{g['predicate']}]--> {g['object']}")                                                                                                                                     
                                                                                                                                                                                                                 
print(f"\n=== Extra triples (not in gold) ===")                                                                                                                                                                    
for i, ext in enumerate(r4_triples):                                                                                                                                                                               
  matched = any(match_triple(ext, g) for g in lu.GOLD_TRIPLES["triples"])                                                                                                                                        
  if not matched:                                       
      print(f"  EXTRA: {ext['subject']} --[{ext['predicate']}]--> {ext['object']}")

# %% [markdown]
# **Discussion:**
# - Did constraining the schema improve or hurt recall?
# - Did it improve precision? Grounding rate?
# - SPOKE uses exactly this approach -- typed edges enforce a schema.
#   What is gained and what is lost?
# - This is the **closed-world assumption**: if it's not in the schema,
#   it can't be extracted. When is this acceptable in clinical AI?

# %% [markdown]
# ---
# ## Round 5: The Specialist (~10 min)
#
# **Unlocked:** Domain-specific fine-tuned models (pre-computed results).
#
# In Rounds 1-4 you used a **general-purpose** LLM and controlled its
# behavior through **prompt engineering**. Now you will examine results
# from two **MedSPO** models -- Qwen2.5 models fine-tuned on 755,000
# PubMed examples specifically for biomedical SPO extraction.
#
# The key difference: MedSPO was *trained on biomedical literature* to
# extract triples. Its extraction knowledge comes from supervised
# fine-tuning on triples distilled from DeepSeek-V3 -- not from a system
# prompt you write at runtime.
#
# This contrast -- **prompt engineering vs. fine-tuning** -- is one of the
# central design decisions in applied AI. Both approaches encode knowledge
# into the system, but in fundamentally different ways.
#
# | Dimension | General LLM + Prompt (R1-4) | MedSPO-3B | MedSPO-7B |
# |-----------|----------------------------|-----------|-----------|
# | How behavior is specified | System prompt at runtime | Trained into weights | Trained into weights |
# | Base model | Llama 3.1 8B | Qwen2.5 3B | Qwen2.5 7B |
# | Training data | General web text | 755k PubMed | 755k PubMed |
# | Schema flexibility | High -- any predicate | Fixed at training | Fixed at training |
# | Output richness | Bare triples | Entity types, confidence | Entity types, confidence |
#
# Both MedSPO models were trained on **identical data** (755k PubMed
# examples). The only difference is the base model size: 3B vs. 7B
# parameters. This lets us isolate the effect of model capacity.

# %% [markdown]
# ### 5.1 Load Pre-Computed Results

# %%
r5_3b_triples = lu.load_medspo_precomputed("3b", "ckd")
r5_7b_triples = lu.load_medspo_precomputed("7b", "ckd")

# %% [markdown]
# ### 5.2 Score Both Models

# %%
r5_3b_groundings = lu.ground_entities(r5_3b_triples)
r5_7b_groundings = lu.ground_entities(r5_7b_triples)

r5_3b_result = lu.show_triple_score(
    r5_3b_triples, lu.GOLD_TRIPLES["triples"],
    label="Round 5a -- MedSPO-3B",
    groundings=r5_3b_groundings,
)

r5_7b_result = lu.show_triple_score(
    r5_7b_triples, lu.GOLD_TRIPLES["triples"],
    label="Round 5b -- MedSPO-7B",
    groundings=r5_7b_groundings,
)

# Submit the better score
r5_best = (r5_3b_result if r5_3b_result["composite"] >= r5_7b_result["composite"]
           else r5_7b_result)
lu.submit_to_leaderboard(5, r5_best)

# %% [markdown]
# ### 5.3 Three-Way Comparison

# %%
print("=== General LLM vs. Fine-Tuned Specialists ===\n")
print(f"{'Metric':<25} {'Your Prompt':>15} {'MedSPO-3B':>15} {'MedSPO-7B':>15}")
print("-" * 72)
for metric, label in [
    ("composite", "Composite"),
    ("triple_f1", "Triple F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("matched", "Matched"),
    ("extra", "Extra (noise)"),
    ("grounding_rate", "Grounding Rate"),
]:
    v4 = r4_result.get(metric, 0)
    v3b = r5_3b_result.get(metric, 0)
    v7b = r5_7b_result.get(metric, 0)
    if metric == "grounding_rate":
        print(f"  {label:<23} {v4:>15.0%} {v3b:>15.0%} {v7b:>15.0%}")
    elif isinstance(v4, int):
        print(f"  {label:<23} {v4:>15} {v3b:>15} {v7b:>15}")
    else:
        print(f"  {label:<23} {v4:>15.3f} {v3b:>15.3f} {v7b:>15.3f}")

# %%
# Inspect MedSPO output -- note the richer metadata
for tag, triples in [("MedSPO-3B", r5_3b_triples), ("MedSPO-7B", r5_7b_triples)]:
    df = pd.DataFrame(triples)
    if not df.empty:
        print(f"\nSample {tag} triples:")
        cols = ["subject", "predicate", "object"]
        if "subject_type" in df.columns:
            cols.extend(["subject_type", "object_type"])
        if "confidence" in df.columns:
            cols.append("confidence")
        display(df[cols].head(10))

# %% [markdown]
# **Discussion:**
# - Where did MedSPO outperform your prompt-engineered pipeline? Where
#   did it fall short?
# - Both MedSPO models were trained on **identical data** (755k PubMed
#   examples). The 7B model has more than twice the parameters of the 3B.
#   Did the extra capacity help? Where?
# - MedSPO returns entity types and confidence scores that your general
#   LLM pipeline does not. What is the value of this metadata for clinical
#   knowledge graphs?
# - Your Round 4 system prompt took ~15 minutes to write. MedSPO's
#   training required 755k curated examples. What are the trade-offs?

# %% [markdown]
# ---
# ## Round 6: New Territory (~12 min)
#
# **The twist:** Extract from a **different article** (Lipid Management)
# using your **exact same** Round 4 system prompt. Also compare against
# pre-computed MedSPO results on the same article.
#
# No changes allowed. This tests **generalization** -- did you build a
# reusable extraction system, or did your rules overfit to the CKD article?
#
# *UTSW has a long history in lipid research -- the institution is home to
# Nobel laureates Michael Brown and Joseph Goldstein, who discovered the
# LDL receptor pathway. The domain is rich in drug-target-outcome triples.*

# %%
LIPID_ARTICLE = (lu.DATA_DIR / "lipids_article.txt").read_text().strip()
LIPID_SENTENCES = [s.strip() for s in LIPID_ARTICLE.split("\n") if s.strip()]
CURVEBALL_GOLD = json.loads(
    (lu.DATA_DIR / "gold_standard_triples_lipids.json").read_text()
)

print(f"Lipid management article: {len(LIPID_SENTENCES)} sentences")
print(f"Curveball gold standard: {len(CURVEBALL_GOLD['triples'])} triples")
print("\nFirst 3 sentences:")
for s in LIPID_SENTENCES[:3]:
    print(f"  {s[:100]}...")

# %% [markdown]
# ### 6.1 General LLM on Lipids

# %%
r6_llm_triples = lu.extract_all_triples(
    LIPID_SENTENCES,
    system_prompt=r4_system_prompt,  # Same prompt as Round 4
    model=lu.SMALL_MODEL,
    temperature=0.0,
)

# %%
r6_llm_groundings = lu.ground_entities(r6_llm_triples)

r6_llm_result = lu.show_triple_score(
    r6_llm_triples, CURVEBALL_GOLD["triples"],
    label="Round 6a -- General LLM on Lipids",
    groundings=r6_llm_groundings,
)

# %% [markdown]
# ### 6.2 MedSPO on Lipids (Pre-Computed)

# %%
r6_3b_triples = lu.load_medspo_precomputed("3b", "lipids")
r6_7b_triples = lu.load_medspo_precomputed("7b", "lipids")

# %%
r6_3b_groundings = lu.ground_entities(r6_3b_triples)
r6_7b_groundings = lu.ground_entities(r6_7b_triples)

r6_3b_result = lu.show_triple_score(
    r6_3b_triples, CURVEBALL_GOLD["triples"],
    label="Round 6b -- MedSPO-3B on Lipids",
    groundings=r6_3b_groundings,
)

r6_7b_result = lu.show_triple_score(
    r6_7b_triples, CURVEBALL_GOLD["triples"],
    label="Round 6c -- MedSPO-7B on Lipids",
    groundings=r6_7b_groundings,
)

# %% [markdown]
# ### 6.3 Generalization Comparison

# %%
# Leaderboard: use the best of three for Round 6
r6_scores = [r6_llm_result, r6_3b_result, r6_7b_result]
r6_best = max(r6_scores, key=lambda r: r["composite"])
lu.submit_to_leaderboard(6, r6_best)

# %%
print("=== Generalization Test: CKD -> Lipids ===\n")
print(f"{'Pipeline':<30} {'CKD':>12} {'Lipids':>12} {'Drop':>10}")
print("-" * 66)
print(f"  {'Your Prompt (R4->R6a)':<28} "
      f"{r4_result['composite']:>12.1f} "
      f"{r6_llm_result['composite']:>12.1f} "
      f"{r4_result['composite'] - r6_llm_result['composite']:>+10.1f}")
print(f"  {'MedSPO-3B (R5a->R6b)':<28} "
      f"{r5_3b_result['composite']:>12.1f} "
      f"{r6_3b_result['composite']:>12.1f} "
      f"{r5_3b_result['composite'] - r6_3b_result['composite']:>+10.1f}")
print(f"  {'MedSPO-7B (R5b->R6c)':<28} "
      f"{r5_7b_result['composite']:>12.1f} "
      f"{r6_7b_result['composite']:>12.1f} "
      f"{r5_7b_result['composite'] - r6_7b_result['composite']:>+10.1f}")

# %% [markdown]
# **Discussion:**
# - Which approach generalized better to a new domain?
# - The general LLM reads your system prompt fresh each time -- it has no
#   "memory" of CKD. MedSPO's weights were frozen at training time. How
#   does this difference affect generalization?
# - MedSPO was trained on PubMed -- both CKD and lipid management are
#   well-represented there. Would you expect the same performance on a
#   less-studied disease?
# - Did the 7B model generalize better than the 3B? What does this
#   tell you about model capacity vs. training data?

# %% [markdown]
# ---
# ## Final Leaderboard and Reflection (~12 min)
#
# Your instructor will display the final cumulative leaderboard.

# %% [markdown]
# ### The Neurosymbolic Pipeline You Built
#
# Over six rounds, you constructed and compared hybrid AI systems:
#
# ```
# Clinical text (natural language)
#        |
#        |-- [General LLM + system prompt rules] --+
#        |                                         |
#        |-- [MedSPO-3B -- domain fine-tuned] -----+
#        |                                         |
#        |-- [MedSPO-7B -- domain fine-tuned] -----+
#        |                                         v
#        |                              Raw SPO triples (structured, but noisy)
#        |                                         |
#        |                              [Gilda -- ontology grounding]
#        |                                         |
#        |                              Grounded triples (standardized IDs)
#        |                                         |
#        |                              [NetworkX -- graph structure]
#        v                                         v
#   Three paths to the same goal:     Knowledge graph (auditable)
#   prompt engineering vs. fine-tuning (3B vs. 7B)
# ```
#
# | Component | Round | What You Learned |
# |-----------|-------|------------------|
# | Bare prompt | 1 | LLMs need explicit structure; bare prompts produce noisy output |
# | System prompt rules | 2 | Knowledge engineering (writing good rules) is hard and valuable |
# | Entity grounding | 3 | Surface forms vary; ontology grounding is essential for consistency |
# | Schema constraint | 4 | Constraining predicates improves precision but may reduce recall |
# | Domain fine-tuned models | 5 | Domain training produces richer output; model size matters |
# | Generalization | 6 | Good rules transfer across domains; overfit rules do not |

# %% [markdown]
# ### Reflection Questions
#
# Discuss in your group and write your responses below.

# %% [markdown]
# **R1: Knowledge Engineering Is the Hard Part**
#
# Compare the effort you spent writing the system prompt (Round 2) to the
# effort of running the extraction. MYCIN's rules were written by hand over
# years. Your "rules" are a system prompt written in minutes. What are the
# trade-offs?

# %%
# YOUR RESPONSE:


# %% [markdown]
# **R2: Grounding and Auditability**
#
# Rank these three systems from most to least auditable for a clinician
# asking "What drugs treat CKD?":
# 1. Your knowledge graph (with source sentences)
# 2. SPOKE (with cited databases)
# 3. A direct LLM query (e.g., ChatGPT)
#
# Explain your ranking.

# %%
# YOUR RESPONSE:


# %% [markdown]
# **R3: Schema vs. Freedom in Clinical AI**
#
# Your Round 4 schema locked predicates to a fixed vocabulary.
# For a hospital deploying a KG extraction system, would you recommend
# the open approach (Round 2) or the locked approach (Round 4)? Why?
# Under what circumstances might the other approach be better?

# %%
# YOUR RESPONSE:


# %% [markdown]
# **R4: Fine-Tuning vs. Prompt Engineering**
#
# You now have two ways to make an LLM extract triples:
# - **Prompt engineering** (Rounds 2-4): write rules at runtime
# - **Fine-tuning** (Round 5): train the rules into the model weights
#
# For each scenario below, which approach would you recommend and why?
# 1. A research team exploring a newly described disease with no established
#    ontology
# 2. A hospital deploying a production system to extract drug interactions
#    from clinical notes at scale
# 3. A student project with limited compute budget and a tight deadline

# %%
# YOUR RESPONSE:


# %% [markdown]
# ---
# ## Submission
#
# Save this notebook and email it to **brian.chapman@utsouthwestern.edu**.
# Ensure:
# - All cells have been run
# - Your group name is set correctly
# - Written reflections (R1-R4) are complete
# - All 6 rounds have been scored and submitted to the leaderboard
