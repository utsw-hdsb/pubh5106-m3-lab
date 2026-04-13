# M3 Lab: Knowledge Engineer

**PUBH 5106 — AI in Health Applications**

A competitive, gamified lab where student teams act as knowledge engineers, extracting biomedical knowledge from clinical text using LLMs and building knowledge graphs.

## Quick Start

1. **Open in Codespaces:** Click the green "Code" button, then "Create codespace on main". Dependencies install automatically (~2 min, includes Gilda grounding database).

2. **Open the notebook:** `M3_student_rev5.ipynb`

3. **Set your group name and API keys** in the Setup section:
   ```python
   lab_utils.GROUP_NAME = "your-group-name"
   lab_utils.set_api_keys(["gsk_...", "gsk_...", "gsk_..."])
   ```
   Each team member needs a free Groq account at [console.groq.com](https://console.groq.com).

4. **Run cells in order.** Six scored rounds, ~2 hours.

## Rounds

| Round | Name | What You Do |
|-------|------|-------------|
| --- | SPOKE Warm-Up | Explore a professional knowledge graph |
| 1 | Cold Extract | Bare-minimum prompt, no rules |
| 2 | Knowledge Engineer | Design a system prompt with extraction rules |
| 3 | Ground Truth | Add entity grounding with Gilda |
| 4 | Schema Lock | Constrained entity types + predicates |
| 5 | The Specialist | Compare fine-tuned MedSPO-3B & 7B models |
| 6 | New Territory | Generalization test on a new article |

## File Layout

```
.
├── M3_student_rev5.ipynb   # Lab notebook
├── M3_student_rev5.py      # Notebook source (jupytext)
├── lab_utils.py            # Shared utilities
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .devcontainer/
│   └── devcontainer.json   # Codespaces configuration
└── data/
    ├── ckd_article.txt
    ├── gold_standard_triples.json
    ├── lipids_article.txt
    ├── gold_standard_triples_lipids.json
    ├── medspo_3b_ckd_precomputed.json
    ├── medspo_3b_lipids_precomputed.json
    ├── medspo_7b_ckd_precomputed.json
    └── medspo_7b_lipids_precomputed.json
```

## Submission

Email your completed notebook to **brian.chapman@utsouthwestern.edu**.
