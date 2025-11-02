#!/usr/bin/env python

import curses
import textwrap
import requests
import argparse
import json
import getpass
import time
import threading
import difflib
import sys
import os
import re
import tempfile
import subprocess
from copy import deepcopy
import concurrent.futures
import yaml
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from models import SpacyModel, LlamaModel, CopyEditor, GrammarlyModel, extractonlytext_sendtollm, get_processed_edits, ConsistencyFindingCoverage, ConsistencyResult, _simple_tokens, _spacy_nouns_phrases, _extract_finding_facets, _extract_summary_facets, _jaccard, _max_sev, _severity_alignment, _SEV_ORDER, _SEV_RANK, _STOP, _WORD

def _toast(stdscr, msg: str, *, row: int = 0, duration_ms: int = 1200):
    """Show a transient message (non-interactive) for duration_ms."""
    try:
        max_y, max_x = stdscr.getmaxyx()
        stdscr.move(row, 0); stdscr.clrtoeol()
        stdscr.addnstr(row, 2, msg, max_x - 4, curses.A_BOLD)
        stdscr.refresh()
        time.sleep(duration_ms / 1000.0)
    except curses.error:
        pass

def _replace_placeholders(s: str, ctx: dict) -> str:
    out = s
    for k, v in (ctx or {}).items():
        out = out.replace(f"{{{{{k}}}}}", str(v or ""))
    return out


def _execsum_postprocess(text: str) -> str:
    if not text: return ""
    t = text.strip()
    for lead in ("Here is", "Here’s", "The following", "Output:", "Draft:", "Summary:"):
        if t.lower().startswith(lead.lower()):
            t = t.split("\n", 1)[-1].strip()
    if t.startswith("```"):
        t = t.strip("` \n")
        if "\n" in t:
            t = t.split("\n", 1)[-1].strip()
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t

def _execsum_scrub_ids(text: str) -> str:
    t = re.sub(r"\bCWE-\d+\b", "", text or "")
    t = re.sub(r"\bCVE-\d{4}-\d+\b", "", t)
    t = re.sub(r"\b(Finding\s*#?\s*\d+|F-\d+)\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

def _finding_row(f: dict) -> str:
    """Return: 'id | title | severity | one-line gist' for prompts."""
    fid = str(f.get("id") or f.get("finding_id") or f.get("flaw_id") or "")
    sev = (f.get("severity") or f.get("Severity") or "").strip().lower() or "unknown"
    title = (f.get("title") or "Untitled").strip()
    desc = (f.get("description") or "").strip().replace("\n", " ")
    gist = (desc[:140] + "…") if len(desc) > 140 else desc

    return f"{fid} | {title} | {sev} | {gist}"


def _concat_exec_summary_from_pages(pages) -> str:
    """Join the paginated exec summary (list[str]) into one string safely."""
    if isinstance(pages, list):
        return "\n\n".join((s or "").strip() for s in pages if isinstance(s, str) and s.strip())
    # fallback if some path ever returns a plain string
    if isinstance(pages, str):
        return pages.strip()
    return ""


def _focus_changed_span(old: str, new: str) -> tuple[str, str]:
    """
    Trim common prefix/suffix at word boundaries so we highlight only the changed core.
    Returns (old_core, new_core). If nothing to trim, returns originals.
    """
    import re
    old_words = re.findall(r'\S+|\s+', old)
    new_words = re.findall(r'\S+|\s+', new)

    # trim prefix
    i = 0
    while i < len(old_words) and i < len(new_words) and old_words[i] == new_words[i]:
        i += 1
    # trim suffix
    j = 0
    while (j < len(old_words) - i) and (j < len(new_words) - i) and \
          old_words[-(j+1)] == new_words[-(j+1)]:
        j += 1

    old_core = ''.join(old_words[i: len(old_words)-j if j else None])
    new_core = ''.join(new_words[i: len(new_words)-j if j else None])

    # fallbacks
    if not old_core and not new_core:
        old_core, new_core = old, new
    return old_core, new_core

def _split_lines_htmlaware(s: str) -> list[str]:
    """
    Convert simple HTML block/line tags to '\n' so context windows align with what the diff view shows.
    We *do not* escape other HTML; display functions already draw plain text.
    """
    if not s:
        return [""]
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Turn block/line separators into newlines
    s = re.sub(r'</p\s*>', '\n', s, flags=re.I)
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
    # Drop opening <p ...> tags
    s = re.sub(r'<p[^>]*>', '', s, flags=re.I)
    # Keep blank lines (they help readability in the viewer)
    return [line.rstrip() for line in s.split('\n')]

def _split_sentences_for_diff(text: str) -> list[str]:
    # Simple, stable splitter (matches what you already use in generate_visual_reportdiff)
    # Keeps sentence punctuation; tolerates whitespace.
    return [s for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if s]

def _make_full_chunk(label: str, original: str, suggested: str) -> list[tuple[str, str]]:
    """
    Build a full, scrollable diff chunk for the entire section.
    Output matches your renderer: list of (kind, text) where kind in
    {'section','add','remove','normal'}.
    """
    chunk = [("section", f"{label}:")]
    a = _split_sentences_for_diff(original)
    b = _split_sentences_for_diff(suggested)
    for line in ndiff(a, b):
        tag, txt = line[:2], line[2:]
        if tag == "+ ":
            chunk.append(("add", txt))
        elif tag == "- ":
            chunk.append(("remove", txt))
        else:
            chunk.append(("normal", txt))
    return chunk

# -------------------- Executive Fitness & Alignment (Shift-C) --------------------

# Minimal stopword set (keep small/fast)
_EF_STOP = set("""
a an the of and or to for from into with without not this that these those be is are was were been being
have has had by on in as at which who whom whose it its they them we you i our your their
""".split())

_EF_WORD = re.compile(r"[A-Za-z0-9_.:/-]{2,}")
_EF_SENT = re.compile(r"[.!?]+(?:\s+|$)")

# Severity order for alignment
_EF_SEV = ["critical", "high", "medium", "low", "informational"]
_EF_RANK = {s: i for i, s in enumerate(_EF_SEV)}

# Theme taxonomy (coarse buckets → synonyms)
_EF_THEME_MAP: Dict[str, List[str]] = {
    "auth": ["auth", "login", "jwt", "oauth", "oidc", "mfa", "session", "csrf", "sso"],
    "data_exposure": ["pii", "phi", "exposure", "leak", "public", "indexing", "bucket", "s3", "listing"],
    "config": ["misconfig", "default", "insecure", "cors", "headers", "tls", "ssl", "hsts"],
    "injection": ["sqli", "xss", "ssti", "injection", "deserialization", "eval"],
    "cloud_perms": ["iam", "role", "policy", "assume", "sts", "privilege", "kms", "vault"],
    "secrets": ["secret", "token", "key", "password", "credential", "hardcoded", ".env"],
    "infra": ["port", "ssh", "rce", "lfi", "rfi", "k8s", "kubernetes", "docker", "container"],
}

# Lexicons for sub-scores
_EF_SYNTHESIS_CUES = {"overall","aggregate","in aggregate","pattern","patterns","themes","trend","trends","therefore","consequently","as a result","in practice","this means"}
_EF_ENUM_CUES = [r"\bFinding\s*#?\b", r"\bCWE-\d+\b", r"\bCVE-\d{4}-\d+\b", r"\bID[:#]?\b", r"\bF-\d+\b"]
_EF_ACTION_VERBS = {"prioritize","remediate","patch","enforce","segment","rotate","monitor","alert","harden","restrict","isolate","decommission","validate","sanitize","encode","rate-limit","migrate","encrypt"}
_EF_PRIORITY_CUES = {"immediately","first","near-term","short-term","phase","roadmap","owner","sla","deadline"}
_EF_HEDGES = {"should consider","might","may","could","appears","likely","possibly","potentially"}
_EF_IMPACT = {"customer","user","data","pii","phi","availability","outage","fraud","financial","revenue","regulatory","sox","hipaa","gdpr","brand","trust","compliance"}

def _ef_tokens(text: str) -> List[str]:
    toks = [t.lower() for t in _EF_WORD.findall(text or "")]
    return [t for t in toks if t not in _EF_STOP]

def _ef_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text: return []
    # naive split, but stable
    out, start = [], 0
    for m in _EF_SENT.finditer(text):
        out.append(text[start:m.end()].strip())
        start = m.end()
    if start < len(text):
        out.append(text[start:].strip())
    return [s for s in out if s]

def _ef_max_sev(values: List[str]) -> str:
    vals = [v for v in values if v in _EF_RANK]
    if not vals: return ""
    return min(vals, key=lambda s: _EF_RANK[s])  # smaller index = higher severity

def _ef_severity_alignment(summary_text: str, finding_severities: List[str]) -> Tuple[bool,str,str]:
    st = summary_text.lower()
    summary_sevs = [s for s in _EF_SEV if s in st]
    s_max = _ef_max_sev(summary_sevs)
    f_max = _ef_max_sev([s.lower() for s in finding_severities if s])
    if not f_max:  # no severities recorded in findings → treat as aligned
        return True, s_max or "", f_max or ""
    if not s_max:
        return False, "", f_max
    return _EF_RANK[s_max] <= _EF_RANK[f_max], s_max, f_max

def _ef_concat_pages(pages_or_str) -> str:
    if isinstance(pages_or_str, list):
        return "\n\n".join((s or "").strip() for s in pages_or_str if isinstance(s, str) and s.strip())
    if isinstance(pages_or_str, str):
        return pages_or_str.strip()
    return ""

def _ef_findings_tokens(findings: List[dict]) -> Tuple[List[str], Counter]:
    all_tokens = []
    for f in findings or []:
        base = " ".join([
            f.get("title",""), f.get("description",""), f.get("recommendations",""),
            ((f.get("fields",{}) or {}).get("guidance",{}) or {}).get("value",""),
            ((f.get("fields",{}) or {}).get("reproduction_steps",{}) or {}).get("value",""),
        ])
        all_tokens.extend(_ef_tokens(base))
        for tag in f.get("tags") or []:
            all_tokens.append(str(tag).lower())
    return all_tokens, Counter(all_tokens)

def _ef_theme_histogram(findings: List[dict]) -> Counter:
    all_tokens, _ = _ef_findings_tokens(findings)
    counts = Counter()
    tokset = set(all_tokens)
    for theme, syns in _EF_THEME_MAP.items():
        for s in syns:
            if s in tokset:
                counts[theme] += 1
                break
    return counts

def _ef_summary_theme_hits(summary_text: str) -> set:
    stoks = set(_ef_tokens(summary_text))
    hits = set()
    for theme, syns in _EF_THEME_MAP.items():
        for s in syns:
            if s in stoks:
                hits.add(theme); break
    return hits

def _ef_synthesis_vs_enumeration(summary_text: str) -> Tuple[int, Dict]:
    txt = summary_text.lower()
    # enumeration indicators
    enum_weight = 0
    bullet_lines = sum(1 for line in summary_text.splitlines() if line.strip().startswith(("-", "*", "•")))
    enum_weight += min(1.0, bullet_lines/6.0)
    for pat in _EF_ENUM_CUES:
        if re.search(pat, txt):
            enum_weight += 0.3
    # synthesis indicators
    synth_hits = sum(1 for cue in _EF_SYNTHESIS_CUES if cue in txt)
    score = max(0, min(100, int(100*(1 - min(1.0, enum_weight)) + 8*synth_hits)))
    return score, {"bullet_lines": bullet_lines, "enum_matches": [p for p in _EF_ENUM_CUES if re.search(p, txt)], "synthesis_hits": synth_hits}

def _ef_business_impact(summary_text: str) -> Tuple[int, Dict]:
    stoks = set(_ef_tokens(summary_text))
    hits = [w for w in _EF_IMPACT if w in stoks]
    base = 40 if hits else 0
    score = max(0, min(100, base + 10*len(hits)))
    return score, {"impact_terms_found": hits[:10]}

def _ef_recommendations(summary_text: str) -> Tuple[int, Dict]:
    txt = summary_text.lower()
    verbs = [v for v in _EF_ACTION_VERBS if re.search(rf"\b{re.escape(v)}\b", txt)]
    prio  = [p for p in _EF_PRIORITY_CUES if re.search(rf"\b{re.escape(p)}\b", txt)]
    hedges = [h for h in _EF_HEDGES if h in txt]
    score = max(0, min(100, 12*len(verbs) + 10*len(prio) - 8*len(hedges)))
    return score, {"action_verbs": verbs[:12], "priority_cues": prio[:12], "hedges": hedges[:12]}

def _ef_readability(summary_text: str) -> Tuple[int, Dict]:
    sents = _ef_sentences(summary_text)
    words = _EF_WORD.findall(summary_text)
    if not sents or not words:
        return 60, {"avg_sentence_len": 0.0, "acronym_density": 0.0}
    avg_len = len(words)/len(sents)
    # heuristic sweet spot 12–24 words
    length_score = 100 - int(abs(avg_len-18) * 5)  # falls off outside 12–24
    # acronym density (ALLCAPS words >= 2 chars, not at sentence start)
    acros = [w for w in re.findall(r"\b[A-Z]{2,}\b", summary_text) if w not in {"TLS","SSL","SSO","JWT","CWE","CVE","S3","IAM"}]
    density = len(acros)/max(1,len(words))
    jargon_penalty = int(400 * density)  # small unless dense
    score = max(0, min(100, length_score - jargon_penalty))
    return score, {"avg_sentence_len": round(avg_len,1), "acronym_density": round(density,3), "acronyms": acros[:10]}


def _english_test_types(selected: list[str]) -> str:
    # Produces: "an external network", "an internal network",
    # "both an external and internal network", "external, internal, and webapp"
    mapping = {
        "external": "external network",
        "internal": "internal network",
        "webapp": "web application",
        "mobile": "mobile application",
    }
    parts = [mapping.get(t, t) for t in selected or []]
    if not parts:
        return "a network and application"  # generic
    if len(parts) == 1:
        return f"an {parts[0]}"
    if len(parts) == 2:
        return f"both an {parts[0]} and {parts[1]}"
    # Oxford comma for >=3
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def merge_exec_fields_for_views(report_content: dict, suggestions: dict | None) -> list[dict]:
    """
    Overlay suggestions onto original exec-summary fields for viewing/diffing.
    - Originals: report_content["exec_summary"]["custom_fields"]
    - Suggestions: suggestions["executive_summary_custom_fields"]["custom_fields"]
    - Match by id first, then by normalized label (fallback).
    - If no originals but suggestions exist, return suggestions so UI still renders.
    """
    def _norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    # Originals come from the report payload
    escf_orig = (
        ((report_content or {}).get("exec_summary") or {}).get("custom_fields")
        or []
    )

    # Suggestions come from the suggested stash
    escf_sugg = (
        ((suggestions or {}).get("executive_summary_custom_fields") or {}).get("custom_fields")
        or []
    )

    # If report has no exec fields yet, still render suggestions
    if not escf_orig and escf_sugg:
        return list(escf_sugg)

    by_id    = {s.get("id"): s for s in escf_sugg if s and s.get("id")}
    by_label = {_norm(s.get("label")): s for s in escf_sugg if s and s.get("label")}

    merged = []
    for f in escf_orig:
        fid   = f.get("id")
        flbl  = _norm(f.get("label"))
        text  = f.get("text", "")

        s = by_id.get(fid) if fid else None
        if s is None:
            s = by_label.get(flbl)

        if s:
            proposed = s.get("text")
            if isinstance(proposed, str) and proposed.strip():
                text = proposed

        merged.append({**f, "text": text})
    return merged

def assess_exec_summary_executive_fitness(exec_text: str, findings: List[dict]) -> dict:
    # Sub-scores
    syn_score, syn_meta = _ef_synthesis_vs_enumeration(exec_text)
    theme_hist = _ef_theme_histogram(findings)
    top_themes = [t for t,_ in theme_hist.most_common(5)]
    hits = _ef_summary_theme_hits(exec_text)
    theme_score = int(100 * (len(set(top_themes[:3]) & hits) / max(1, min(3, len(top_themes))))) if top_themes else 0
    theme_meta = {"top_themes": top_themes, "mentioned": sorted(hits), "missing": [t for t in top_themes if t not in hits]}

    impact_score, impact_meta = _ef_business_impact(exec_text)
    reco_score, reco_meta = _ef_recommendations(exec_text)
    read_score, read_meta = _ef_readability(exec_text)

    # Severity alignment
    sev_list = [ (f.get("severity") or f.get("Severity") or "").lower() for f in (findings or []) ]
    sev_ok, s_max, f_max = _ef_severity_alignment(exec_text, sev_list)
    sev_penalty = 20 if not sev_ok else 0

    # Final score (weights emphasize exec tone + impact)
    final = int(max(0, min(100,
        0.25*syn_score + 0.20*theme_score + 0.20*impact_score + 0.20*reco_score + 0.10*read_score - sev_penalty
    )))

    return {
        "score": final,
        "severity_aligned": sev_ok,
        "summary_max_sev": s_max or "-",
        "findings_max_sev": f_max or "-",
        "sections": {
            "synthesis": {"score": syn_score, **syn_meta},
            "themes": {"score": theme_score, **theme_meta},
            "impact": {"score": impact_score, **impact_meta},
            "recommendations": {"score": reco_score, **reco_meta},
            "readability": {"score": read_score, **read_meta},
        }
    }



def assess_exec_summary_consistency(exec_text: str, findings: list[dict]) -> ConsistencyResult:
    s_facets = _extract_summary_facets(exec_text or "")
    f_facets = [_extract_finding_facets(f) for f in (findings or [])]

    coverage: Dict[str, ConsistencyFindingCoverage] = {}
    overlaps = []
    covered_ids, uncovered_ids = [], []

    # threshold can be tuned; summaries are short, so keep modest
    THRESH = 0.22

    for f, facet in zip(findings or [], f_facets):
        key = str(f.get("id") or f.get("finding_id") or f.get("title") or f"idx:{findings.index(f)}")
        score = _jaccard(facet["tokens"], s_facets["tokens"])
        covered = score >= THRESH
        if covered:
            covered_ids.append(key)
        else:
            uncovered_ids.append(key)

        overlaps.append(score)
        # tiny “evidence” = top overlapping tokens
        common = list((set(facet["tokens"]) & set(s_facets["tokens"])))
        evidence = ", ".join(sorted(common, key=len, reverse=True)[:6])
        coverage[key] = ConsistencyFindingCoverage(covered=covered, score=round(score, 3), evidence=evidence)

    sev_ok, s_max, f_max = _severity_alignment(s_facets["severities"], [ff["severity"] for ff in f_facets])

    # rule-based contradictions
    contradictions = []
    if f_max in {"critical", "high"} and (not s_max or _SEV_RANK.get(s_max, 99) > _SEV_RANK[f_max]):
        contradictions.append("Summary downplays highest severity present in findings")

    # crude themed gaps: cluster by most frequent tokens per finding not present in summary
    token_counts = Counter(t for ff in f_facets for t in ff["tokens"])
    top_tokens = [t for t, _ in token_counts.most_common(50)]
    sum_tokens = set(s_facets["tokens"])
    gaps_map = defaultdict(list)
    for f, ff in zip(findings or [], f_facets):
        if str(f.get("id") or f.get("finding_id") or f.get("title")) in covered_ids:
            continue
        # tokens strong for this finding that are missing from summary
        missing = [t for t in ff["tokens"] if t in top_tokens and t not in sum_tokens]
        if missing:
            gaps_map[missing[0]].append(str(f.get("title") or f.get("id") or f.get("finding_id")))

    theme_gaps = [{"theme": k, "example_findings": v[:5]} for k, v in gaps_map.items()]

    # confidence score (0..100)
    R = (len(covered_ids) / max(1, len(findings or []))) if findings else 0.0
    mean_overlap = (sum(overlaps) / len(overlaps)) if overlaps else 0.0
    sev_pen = 0 if sev_ok else 20
    contr_pen = min(30, 10 * len(contradictions))
    score = int(max(0, min(100, 100 * (0.5 * R + 0.3 * mean_overlap) - sev_pen - contr_pen)))

    return ConsistencyResult(
        coverage=coverage,
        covered_ids=covered_ids,
        uncovered_ids=uncovered_ids,
        severity_alignment_ok=sev_ok,
        summary_max_sev=s_max or "-",
        findings_max_sev=f_max or "-",
        contradictions=contradictions,
        theme_gaps=theme_gaps,
        confidence=score,
        summary_tokens=s_facets["tokens"][:50],
        summary_severities=s_facets["severities"],
    )


def is_executive_summary_unchanged(api, narrative_id):
    """
    Finds and compares the original and modified executive summary narratives for a given ID.

    :param narrative_id: ID of the narrative text to check.
    :return: True if unchanged, False if modified.
    """
    # Find the original narrative by ID
    original_narrative = next((item for item in api.report_content["exec_summary"]["custom_fields"] if item.get("id") == narrative_id), None)

    # Find the modified narrative by ID
    modified_narrative = next((item for item in api.suggestedfixes_from_llm["executive_summary_custom_fields"]["custom_fields"] if item.get("id") == narrative_id), None)

    # If either is missing, consider them different
    if original_narrative is None or modified_narrative is None:
        return False

    # Compare the text fields
    return original_narrative.get("text", "") == modified_narrative.get("text", "")

def is_finding_title_unchanged(api, finding_id):
    """
    Finds and compares the original and modified finding titles for a given ID.

    :param api: API instance containing report data.
    :param finding_id: ID of the finding to check.
    :return: True if unchanged, False if modified.
    """
    original_finding = next((item for item in api.report_findings_content if item.get("flaw_id") == finding_id), None)
    modified_finding = next((item for item in api.suggestedfixes_from_llm["findings"] if item.get("flaw_id") == finding_id), None)

    if original_finding is None or modified_finding is None:
        return False

    return original_finding.get("title", "") == modified_finding.get("title", "")

def is_finding_description_unchanged(api, finding_id):
    """
    Finds and compares the original and modified finding descriptions for a given ID.

    :param api: API instance containing report data.
    :param finding_id: ID of the finding to check.
    :return: True if unchanged, False if modified.
    """
    original_finding = next((item for item in api.report_findings_content if item.get("flaw_id") == finding_id), None)
    modified_finding = next((item for item in api.suggestedfixes_from_llm["findings"] if item.get("flaw_id") == finding_id), None)

    if original_finding is None or modified_finding is None:
        return False

    return original_finding.get("description", "") == modified_finding.get("description", "")


def is_finding_recommendations_unchanged(api, finding_id):
    """
    Finds and compares the original and modified finding recommendations for a given ID.

    :param api: API instance containing report data.
    :param finding_id: ID of the finding to check.
    :return: True if unchanged, False if modified.
    """
    original_finding = next((item for item in api.report_findings_content if item.get("flaw_id") == finding_id), None)
    modified_finding = next((item for item in api.suggestedfixes_from_llm["findings"] if item.get("flaw_id") == finding_id), None)

    if original_finding is None or modified_finding is None:
        return False

    return original_finding.get("recommendations", "") == modified_finding.get("recommendations", "")


def is_finding_guidance_unchanged(api, finding_id):
    """
    Finds and compares the original and modified finding guidance for a given ID.

    :param api: API instance containing report data.
    :param finding_id: ID of the finding to check.
    :return: True if unchanged, False if modified.
    """
    original_finding = next((item for item in api.report_findings_content if item.get("flaw_id") == finding_id), None)
    modified_finding = next((item for item in api.suggestedfixes_from_llm["findings"] if item.get("flaw_id") == finding_id), None)

    if original_finding is None or modified_finding is None:
        return False

    return original_finding.get("fields","").get("guidance", {}).get("value","")==modified_finding.get("fields","").get("guidance", {}).get("value","")

def is_finding_reproduction_steps_unchanged(api, finding_id):
    """
    Finds and compares the original and modified finding reproduction_steps for a given ID.

    :param api: API instance containing report data.
    :param finding_id: ID of the finding to check.
    :return: True if unchanged, False if modified.
    """
    original_finding = next((item for item in api.report_findings_content if item.get("flaw_id") == finding_id), None)
    modified_finding = next((item for item in api.suggestedfixes_from_llm["findings"] if item.get("flaw_id") == finding_id), None)

    if original_finding is None or modified_finding is None:
        return False

    return original_finding.get("fields","").get("reproduction_steps", {}).get("value","") == modified_finding.get("fields","").get("reproduction_steps", {}).get("value","")

