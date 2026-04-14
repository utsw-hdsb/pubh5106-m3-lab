"""M3 Lab Utilities: The Knowledge Engineer

Infrastructure code for the M3 knowledge graph extraction lab. This module
provides:
- LLM interaction via Groq API (with API key rotation)
- Triple parsing (general LLM format)
- Sentence-by-sentence extraction pipeline
- Pre-computed MedSPO-3B and MedSPO-7B triple loading
- Scoring (fuzzy match, keyword match, triple F1)
- Entity grounding via Gilda
- Composite scoring and display
- Leaderboard submission
- Data loading helpers

Students import this module and focus on writing system prompts, analyzing
extraction results, and building knowledge graphs.
"""

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import gilda
import requests
from IPython.display import display, Markdown

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")

# Backend: "groq" (default, for students) or "ollama" (for local testing)
BACKEND = "groq"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_API_URL = "http://localhost:11434"

SMALL_MODEL = "llama-3.1-8b-instant"

# Ollama model name (used when BACKEND == "ollama")
OLLAMA_MODEL = "llama3.1:8b"

# Leaderboard (Google Forms) — instructor configures before class
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdE1zpZDv9whj3poTtQvTyXe9Z72B0FJDTAJA-db0ZfSt1O5A/formResponse"
FIELD_GROUP = "entry.580574446"
FIELD_ROUND = "entry.2102634743"
FIELD_SCORE = "entry.1233314305"
FIELD_DETAIL = "entry.1762914451"

# Group name — students set this in the notebook via: lab_utils.GROUP_NAME = "..."
GROUP_NAME = "CHANGE_ME"

# ─────────────────────────────────────────────────────────────────────
# API Key Management
# ─────────────────────────────────────────────────────────────────────

_api_keys = []        # List of Groq API keys (set by students in notebook)
_current_key_idx = 0  # Index of the key currently in use


def set_api_keys(keys: list[str]) -> None:
    """Register one or more Groq API keys for the session.

    Each team member should create a free Groq account and generate an
    API key. Enter all keys here — the system will automatically rotate
    to the next key if the current one hits its free-tier rate limit.

    Parameters
    ----------
    keys : list[str]
        A list of Groq API key strings.
    """
    global _api_keys, _current_key_idx
    _api_keys = [k.strip() for k in keys if k.strip()]
    _current_key_idx = 0
    print(f"Registered {len(_api_keys)} API key(s)")
    if not _api_keys:
        print("  *** No valid keys provided! ***")


def _get_current_key() -> str:
    """Return the current API key."""
    if not _api_keys:
        raise RuntimeError(
            "No API keys registered. Call set_api_keys() first.\n"
            "Example: lab_utils.set_api_keys(['gsk_abc123...', 'gsk_def456...'])"
        )
    return _api_keys[_current_key_idx]


def _rotate_key() -> bool:
    """Rotate to the next API key. Returns False if all keys exhausted."""
    global _current_key_idx
    _current_key_idx += 1
    if _current_key_idx >= len(_api_keys):
        _current_key_idx = 0  # wrap around
        return False  # all keys have been tried
    print(f"  Rotating to API key {_current_key_idx + 1}/{len(_api_keys)}")
    return True


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

try:
    CKD_ARTICLE = (DATA_DIR / "ckd_article.txt").read_text().strip()
    CKD_SENTENCES = [s.strip() for s in CKD_ARTICLE.split("\n") if s.strip()]
except FileNotFoundError:
    CKD_ARTICLE = ""
    CKD_SENTENCES = []
    print("WARNING: ckd_article.txt not found. Set DATA_DIR correctly.")

try:
    GOLD_TRIPLES = json.loads(
        (DATA_DIR / "gold_standard_triples.json").read_text()
    )
except FileNotFoundError:
    GOLD_TRIPLES = {"triples": []}
    print("WARNING: gold_standard_triples.json not found. Set DATA_DIR correctly.")


# ─────────────────────────────────────────────────────────────────────
# Pre-computed MedSPO results
# ─────────────────────────────────────────────────────────────────────

def load_medspo_precomputed(model_size: str, article: str) -> list[dict]:
    """Load pre-computed MedSPO extraction results.

    Parameters
    ----------
    model_size : str
        "3b" or "7b"
    article : str
        "ckd" or "lipids"

    Returns
    -------
    list[dict]
        List of SPO triple dicts with metadata (subject_type, confidence, etc.)
    """
    filename = f"medspo_{model_size}_{article}_precomputed.json"
    path = DATA_DIR / filename
    if path.exists():
        triples = json.loads(path.read_text())
        print(f"Loaded {len(triples)} pre-computed MedSPO-{model_size.upper()} "
              f"triples ({article})")
        return triples

    # Fall back to legacy filename for 3B
    if model_size == "3b":
        legacy = DATA_DIR / f"medspo_{article}_precomputed.json"
        if legacy.exists():
            triples = json.loads(legacy.read_text())
            print(f"Loaded {len(triples)} pre-computed MedSPO-3B triples "
                  f"({article}, legacy file)")
            return triples

    print(f"WARNING: {filename} not found in {DATA_DIR}")
    return []


# ─────────────────────────────────────────────────────────────────────
# LLM interaction
# ─────────────────────────────────────────────────────────────────────

def _call_ollama(messages: list[dict], model: str, temperature: float | None) -> str:
    """Call a local Ollama model."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    t0 = time.time()
    response = requests.post(
        f"{OLLAMA_API_URL}/api/chat",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    elapsed = time.time() - t0
    result = response.json()["message"]["content"]
    print(f"  [{model}] {len(result)} chars, {elapsed:.1f}s (ollama)")
    return result


def _call_groq(messages: list[dict], model: str, temperature: float | None) -> str:
    """Call Groq API with key rotation."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    keys_tried = 0
    max_retries_per_key = 3
    t0 = time.time()

    while keys_tried < len(_api_keys):
        api_key = _get_current_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries_per_key):
            response = requests.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=120,
            )

            if response.status_code == 200:
                elapsed = time.time() - t0
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                print(f"  [{model}] {len(result)} chars, {elapsed:.1f}s "
                      f"(key {_current_key_idx + 1}/{len(_api_keys)}, "
                      f"{usage.get('total_tokens', '?')} tokens)")
                return result

            elif response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                if retry_after and float(retry_after) < 30:
                    wait = float(retry_after) + 1
                    print(f"  Rate limited (key {_current_key_idx + 1}), "
                          f"waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"  Rate limited (key {_current_key_idx + 1}), "
                          f"trying next key...")
                    break

            else:
                response.raise_for_status()

        keys_tried += 1
        if not _rotate_key():
            break

    raise RuntimeError(
        "All API keys have been rate-limited. Wait a few minutes and try again, "
        "or add more API keys with set_api_keys()."
    )


def call_llm(
    user_message: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Call an LLM. Uses BACKEND setting to route to Groq or Ollama."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    if BACKEND == "ollama":
        # Always use OLLAMA_MODEL — ignore Groq model names passed by notebook
        return _call_ollama(messages, OLLAMA_MODEL, temperature)
    else:
        groq_model = model if model else SMALL_MODEL
        return _call_groq(messages, groq_model, temperature)


def extract_all_triples(
    sentences: list[str],
    system_prompt: str | None = None,
    model: str = SMALL_MODEL,
    temperature: float | None = 0.0,
) -> list[dict]:
    """Extract SPO triples from all sentences. Returns flat list of triples."""
    all_triples = []
    for i, sentence in enumerate(sentences, 1):
        user_msg = f"Extract SPO triples from this sentence:\n\n{sentence}"
        raw = call_llm(user_msg, system_prompt=system_prompt,
                       model=model, temperature=temperature)
        triples = parse_llm_triples(raw)
        for t in triples:
            t["source_sentence"] = sentence
        all_triples.extend(triples)
        status = f"{len(triples)} triple(s)" if triples else "none"
        print(f"  [{i:>2}/{len(sentences)}] {status}")
    print(f"\nTotal extracted: {len(all_triples)} triples")
    return all_triples


# ─────────────────────────────────────────────────────────────────────
# Triple parsing
# ─────────────────────────────────────────────────────────────────────

def parse_llm_triples(text: str) -> list[dict]:
    """Extract a JSON array of triples from LLM output."""
    # Try code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    # Find outermost brackets
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            arr = json.loads(bracket_match.group())
            return [t for t in arr
                    if isinstance(t, dict)
                    and all(k in t for k in ("subject", "predicate", "object"))]
        except json.JSONDecodeError:
            pass
    return []


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def fuzzy_match(candidate: str, target: str, threshold: float = 0.55) -> bool:
    """Check if candidate fuzzy-matches target."""
    c = candidate.lower().strip()
    t = target.lower().strip()
    if not c or not t:
        return False
    if c in t or t in c:
        return True
    return SequenceMatcher(None, c, t).ratio() >= threshold


def keyword_match(candidate: str, keywords: list[str]) -> bool:
    """Check if candidate matches any keyword (substring, case-insensitive)."""
    c = candidate.lower()
    return any(kw.lower() in c for kw in keywords)


def _match_directed(ext_triple: dict, gold_triple: dict) -> bool:
    """Check if extracted triple matches gold in the given direction."""
    s_match = (fuzzy_match(ext_triple.get("subject", ""), gold_triple["subject"])
               or keyword_match(ext_triple.get("subject", ""), gold_triple["keywords_s"]))
    if not s_match:
        return False

    o_match = (fuzzy_match(ext_triple.get("object", ""), gold_triple["object"])
               or keyword_match(ext_triple.get("object", ""), gold_triple["keywords_o"]))
    if not o_match:
        return False

    p_match = (fuzzy_match(ext_triple.get("predicate", ""), gold_triple["predicate"])
               or keyword_match(ext_triple.get("predicate", ""), gold_triple["keywords_p"]))
    return p_match


def match_triple(ext_triple: dict, gold_triple: dict) -> bool:
    """Check if an extracted triple matches a gold standard triple.

    Tries both directions: if the LLM reversed subject and object but
    got the right entities and a matching predicate, it still counts.
    This is standard in KG evaluation — directionality errors are
    penalized separately from entity/predicate errors.
    """
    if _match_directed(ext_triple, gold_triple):
        return True

    # Try swapped: extracted subject matches gold object and vice versa
    swapped = {
        "subject": ext_triple.get("object", ""),
        "object": ext_triple.get("subject", ""),
        "predicate": ext_triple.get("predicate", ""),
    }
    return _match_directed(swapped, gold_triple)


def score_triples(extracted: list[dict], gold: list[dict]) -> dict:
    """Score extracted triples against gold standard.

    Returns precision, recall, F1, and match details.
    """
    matched_gold = set()
    matched_ext = set()

    for i, ext in enumerate(extracted):
        for j, g in enumerate(gold):
            if j in matched_gold:
                continue
            if match_triple(ext, g):
                matched_gold.add(j)
                matched_ext.add(i)
                break

    n_matched = len(matched_gold)
    n_gold = len(gold)
    n_ext = len(extracted)
    precision = n_matched / n_ext if n_ext > 0 else 0
    recall = n_matched / n_gold if n_gold > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return {
        "matched": n_matched,
        "missed": n_gold - n_matched,
        "extra": n_ext - len(matched_ext),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "total_extracted": n_ext,
        "total_gold": n_gold,
    }


# ─────────────────────────────────────────────────────────────────────
# Entity grounding
# ─────────────────────────────────────────────────────────────────────

def ground_entities(triples: list[dict]) -> dict:
    """Ground all unique entities from triples using Gilda.

    Returns dict mapping entity string -> grounding result or None.
    """
    entities = set()
    for t in triples:
        entities.add(t.get("subject", "").lower().strip())
        entities.add(t.get("object", "").lower().strip())
    entities.discard("")

    groundings = {}
    for entity in sorted(entities):
        try:
            matches = gilda.ground(entity)
            if matches:
                top = matches[0]
                groundings[entity] = {
                    "label": top.term.entry_name,
                    "ontology": top.term.db,
                    "id": f"{top.term.db}:{top.term.id}",
                    "score": round(top.score, 3),
                }
            else:
                groundings[entity] = None
        except Exception:
            groundings[entity] = None
    return groundings


def grounding_rate(groundings: dict) -> float:
    """Return fraction of entities successfully grounded."""
    if not groundings:
        return 0.0
    matched = sum(1 for v in groundings.values() if v is not None)
    return round(matched / len(groundings), 3)


# ─────────────────────────────────────────────────────────────────────
# Composite scoring and display
# ─────────────────────────────────────────────────────────────────────

def composite_score(
    triple_score: dict,
    g_rate: float = 0.0,
    has_grounding: bool = False,
) -> float:
    """Compute composite score for leaderboard.

    Rounds 1-2: 100% triple F1
    Rounds 3+:  70% triple F1 + 30% grounding rate
    """
    if has_grounding:
        return round((0.70 * triple_score["f1"] + 0.30 * g_rate) * 100, 1)
    return round(triple_score["f1"] * 100, 1)


def show_triple_score(
    triples: list[dict],
    gold: list[dict],
    label: str = "",
    groundings: dict | None = None,
) -> dict:
    """Score, display, and return results."""
    ts = score_triples(triples, gold)
    g_rate = grounding_rate(groundings) if groundings else 0.0
    has_g = groundings is not None
    comp = composite_score(ts, g_rate, has_g)

    parts = [f"Composite: **{comp}** / 100"]
    parts.append(f"Triple F1: {ts['f1']}")
    parts.append(f"Matched: {ts['matched']}/{ts['total_gold']}")
    parts.append(f"Extra: {ts['extra']}")
    if has_g:
        parts.append(f"Grounding: {g_rate:.0%}")
    header = f"**{label}** -- " if label else ""
    display(Markdown(header + " | ".join(parts)))

    return {
        "composite": comp,
        "triple_f1": ts["f1"],
        "precision": ts["precision"],
        "recall": ts["recall"],
        "matched": ts["matched"],
        "missed": ts["missed"],
        "extra": ts["extra"],
        "grounding_rate": g_rate,
    }


# ─────────────────────────────────────────────────────────────────────
# Diagnostic: missed and extra triples
# ─────────────────────────────────────────────────────────────────────

def show_missed_and_extra(extracted: list[dict], gold: list[dict]) -> None:
    """Show which gold triples were missed and which extracted triples
    are not in the gold standard. Helps diagnose extraction errors."""
    print("=== Missed gold triples ===")
    missed = 0
    for g in gold:
        if not any(match_triple(ext, g) for ext in extracted):
            print(f"  MISS: {g['subject']} --[{g['predicate']}]--> {g['object']}")
            missed += 1
    if missed == 0:
        print("  (none — all gold triples matched!)")

    print(f"\n=== Extra triples (not in gold) ===")
    extra = 0
    for ext in extracted:
        if not any(match_triple(ext, g) for g in gold):
            print(f"  EXTRA: {ext.get('subject', '?')} "
                  f"--[{ext.get('predicate', '?')}]--> "
                  f"{ext.get('object', '?')}")
            extra += 1
    if extra == 0:
        print("  (none — perfect precision!)")

    print(f"\nSummary: {len(gold) - missed}/{len(gold)} matched, "
          f"{extra} extra")


# ─────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────

def submit_to_leaderboard(round_num: int, result: dict) -> None:
    """Submit score to Google Forms leaderboard."""
    if "REPLACE" in FORM_URL:
        print(f"[Leaderboard not configured] Round {round_num}: "
              f"composite={result['composite']}")
        return
    try:
        detail = json.dumps({k: result[k] for k in
                             ["triple_f1", "precision", "recall",
                              "matched", "grounding_rate"]})
        requests.post(FORM_URL, data={
            FIELD_GROUP: GROUP_NAME,
            FIELD_ROUND: round_num,
            FIELD_SCORE: result["composite"],
            FIELD_DETAIL: detail,
        }, timeout=10)
        print(f"Submitted: Round {round_num}, Score {result['composite']}")
    except Exception as e:
        print(f"Submission failed: {e}")


# ─────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────

def use_ollama(model: str = "llama3.1:8b", url: str = "http://localhost:11434") -> None:
    """Switch to local Ollama backend for testing.

    Usage in notebook:
        lab_utils.use_ollama()                    # defaults
        lab_utils.use_ollama("llama3.2:3b")       # different model
    """
    global BACKEND, OLLAMA_MODEL, OLLAMA_API_URL
    BACKEND = "ollama"
    OLLAMA_MODEL = model
    OLLAMA_API_URL = url
    print(f"Switched to Ollama backend: {model} at {url}")


def use_groq() -> None:
    """Switch back to Groq backend (default for students)."""
    global BACKEND
    BACKEND = "groq"
    print(f"Switched to Groq backend: {SMALL_MODEL}")


def verify_setup() -> None:
    """Verify that data files are present and the LLM backend is reachable."""
    print("Checking data files...")
    for name in ["ckd_article.txt", "gold_standard_triples.json",
                 "lipids_article.txt", "gold_standard_triples_lipids.json"]:
        path = DATA_DIR / name
        status = "found" if path.exists() else "NOT FOUND"
        print(f"  {name}: {status}")

    precomputed = [
        "medspo_3b_ckd_precomputed.json",
        "medspo_3b_lipids_precomputed.json",
        "medspo_7b_ckd_precomputed.json",
        "medspo_7b_lipids_precomputed.json",
    ]
    legacy = ["medspo_ckd_precomputed.json", "medspo_lipids_precomputed.json"]
    for name in precomputed + legacy:
        path = DATA_DIR / name
        if path.exists():
            print(f"  {name}: found")

    print(f"\nBackend: {BACKEND}")
    if BACKEND == "ollama":
        print(f"  Model: {OLLAMA_MODEL}")
        print(f"  URL: {OLLAMA_API_URL}")
        try:
            r = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            found = any(OLLAMA_MODEL in m for m in models)
            status = "available" if found else "NOT FOUND"
            print(f"  {OLLAMA_MODEL}: {status}")
        except Exception as e:
            print(f"  Ollama not reachable: {e}")
    else:
        if not _api_keys:
            print("  No API keys registered yet.")
            print("  *** Call lab_utils.set_api_keys(['gsk_...']) ***")
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {_get_current_key()}",
                    "Content-Type": "application/json",
                }
                r = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers=headers,
                    timeout=10,
                )
                r.raise_for_status()
                models = [m["id"] for m in r.json().get("data", [])]
                found = SMALL_MODEL in models
                status = "available" if found else "NOT FOUND"
                print(f"  {SMALL_MODEL}: {status}")
                print(f"  {len(_api_keys)} API key(s) registered")
            except Exception as e:
                print(f"  API check failed: {e}")

    print(f"\nGroup: {GROUP_NAME}")
    if GROUP_NAME == "CHANGE_ME":
        print("  *** Set your group name in the notebook: "
              "lab_utils.GROUP_NAME = '...' ***")

    print(f"\nCKD article: {len(CKD_SENTENCES)} sentences, "
          f"{len(CKD_ARTICLE):,} chars")
    print(f"Gold standard: {len(GOLD_TRIPLES.get('triples', []))} triples")
