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


from models import SpacyModel, LlamaModel, CopyEditor, GrammarlyModel, extractonlytext_sendtollm, get_processed_edits, ConsistencyFindingCoverage, ConsistencyResult, _simple_tokens, _spacy_nouns_phrases, _extract_finding_facets, _extract_summary_facets, _jaccard, _max_sev, _severity_alignment, _SEV_ORDER, _SEV_RANK, _STOP, _WORD
from textwrap import shorten
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher, ndiff

from utils import (
        _toast,
        _replace_placeholders,
        _execsum_postprocess,
        _execsum_scrub_ids,
        _finding_row,
        _concat_exec_summary_from_pages,
        _focus_changed_span,
        _split_lines_htmlaware,
        _split_sentences_for_diff,
        _make_full_chunk,
        _ef_tokens,
        _ef_sentences,
        _ef_max_sev,
        _ef_severity_alignment,
        _ef_concat_pages,
        _ef_findings_tokens,
        _ef_theme_histogram,
        _ef_summary_theme_hits,
        _ef_synthesis_vs_enumeration,
        _ef_business_impact,
        _ef_recommendations,
        _ef_readability,
        _EF_STOP,
        _EF_WORD,
        _EF_SENT,
        _EF_SEV,
        _EF_RANK,
        _EF_THEME_MAP,
        _EF_SYNTHESIS_CUES,
        _EF_ENUM_CUES,
        _EF_ACTION_VERBS,
        _EF_PRIORITY_CUES,
        _EF_HEDGES,
        _EF_IMPACT,
        assess_exec_summary_consistency,
        assess_exec_summary_executive_fitness,
        is_executive_summary_unchanged,
        is_finding_title_unchanged,
        is_finding_description_unchanged,
        is_finding_recommendations_unchanged,
        is_finding_guidance_unchanged,
        is_finding_reproduction_steps_unchanged
)


@dataclass
class Hunk:
    """
    A hunk = one atomic change at sentence granularity:

    replace: one original sentence → one suggested sentence
    delete: original sentence removed
    insert: new sentence added

    Each hunk carries:

    hunk_id: a stable-ish string so we can reference/log it
    change_kind: "replace" | "delete" | "insert"
    sent_idx_orig: the anchor index in the original sentence list (for inserts: “insert after this index”)
    original_text and/or suggested_text
    state: "pending" | "accepted" | "edited" | "rejected"
    """
    hunk_id: str
    change_kind: str        # 'replace' | 'insert' | 'delete'
    sent_idx_orig: int      # anchor in original sentences (for inserts, anchor position to insert after)
    original_text: str
    suggested_text: str
    state: str = "pending"  # 'pending' | 'accepted' | 'edited' | 'rejected'


# User Interface 
# ------------------------------------------------------------------------------------------

def soft_format_html_for_terminal(text):
    """
    Adds basic newlines/indents for CLI readability while preserving all HTML tags.
    Does NOT strip or decode any tags — only adds line breaks and minimal formatting.
    """
    # Block-level tags → insert newlines before and/or after
    text = re.sub(r'</?(p|div|section|h[1-6]|br)[^>]*>', r'\n', text, flags=re.IGNORECASE)

    # List tags → format like bullets
    text = re.sub(r'<ul[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</ul>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<ol[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</ol>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li[^>]*>', '\n  - ', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)

    # Blockquotes → prefix with >
    text = re.sub(r'<blockquote[^>]*>', '\n> ', text, flags=re.IGNORECASE)
    text = re.sub(r'</blockquote>', '\n', text, flags=re.IGNORECASE)

    # Add line spacing around code blocks
    text = re.sub(r'<pre[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</pre>', '\n', text, flags=re.IGNORECASE)

    # Normalize line breaks (no more than 2 in a row)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()



def interactive_text_viewer(stdscr, api, pages):
    """Curses-based interactive viewer with text wrapping and vertical scrolling."""
    curses.curs_set(0)
    page_index = 0
    current_view = "ORIGINAL VIEW-"
    scroll_offset = 0  # Track vertical scrolling

    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()

        # Message at the top of the window
        top_message = f"Page {page_index + 1}/{len(pages)} (q-quit, c-gen/regen exec summary, shift+c- verify exec summary, r-get llm suggestions, p-view suggestions, o-view original, d-view diffs, u-import updates to plex)"
        stdscr.addstr(0, 0, current_view + top_message, curses.A_BOLD)

        # Wrap text properly
        raw_page = "\n".join(pages[page_index])
        formatted = soft_format_html_for_terminal(raw_page)
        wrapped_lines = []

        for paragraph in formatted.splitlines():
            wrapped_lines.extend(textwrap.wrap(paragraph, width=max_x - 4))

        total_lines = len(wrapped_lines)
        visible_lines = wrapped_lines[scroll_offset: scroll_offset + max_y - 3]  # Leave space for header

        # Display visible text
        for i, line in enumerate(visible_lines):
            stdscr.addstr(i + 2, 2, line)

        stdscr.refresh()
        key = stdscr.getch()

        # Quit viewer
        if key == ord('q'):
            break
        # Scroll down
        elif key == curses.KEY_DOWN and scroll_offset < total_lines - (max_y - 3):
            scroll_offset += 1
        # Scroll up
        elif key == curses.KEY_UP and scroll_offset > 0:
            scroll_offset -= 1
        # Next page
        elif key in (curses.KEY_RIGHT, ord('l')) and page_index < len(pages) - 1:
            page_index += 1
            scroll_offset = 0  # Reset scrolling on new page
        # Previous page
        elif key in (curses.KEY_LEFT, ord('h')) and page_index > 0:
            page_index -= 1
            scroll_offset = 0  # Reset scrolling on new page
        # Get LLM suggestions
        elif key == ord('r'):
            stdscr.addstr(1, 0, "Sending report to LLM for suggestions...", curses.A_BOLD)
            stdscr.refresh()
            api.get_suggested_fixes_from_llm(use_llama=api.use_llama, use_grammarly=api.use_grammarly)
            api.generate_visual_reportdiff()
        # View LLM suggestions
        elif key == ord('p') and api.retrieved_suggestedfixes_from_llm:
            pages = generate_paginated_text(api.get_local_exec_summary(original=False) + api.get_local_findings(original=False))
            current_view = "LLM VIEW-"
            page_index = 0
            scroll_offset = 0
        # View Diff mode
        elif key == ord('d') and api.visual_diff_generated:
            display_visual_diff_mode(stdscr, api.visual_diff)
        # View original report
        elif key == ord('o'):
            pages = generate_paginated_text(api.get_local_exec_summary(original=True) + api.get_local_findings(original=True))
            current_view = "ORIGINAL VIEW-"
            page_index = 0
            scroll_offset = 0
        # Import updates into PlexTrac
        elif key == ord('u') and api.retrieved_suggestedfixes_from_llm:
            import_llm_suggestions(api, stdscr, max_x, max_y)
        elif key == ord('C'):  # shift+c = consistency check report
            show_exec_summary_fitness_report(stdscr, api)
            #show_exec_summary_consistency_report(stdscr, api)
        # Use LLM to generate executive summary from the findings list
        elif key == ord('c'):
            chosen = pick_test_types(stdscr, api.exec_summary_ctx.get("selected_types"))
            if chosen is None:
                continue
            if not chosen:
                _toast(stdscr, "No test types selected — cancelled.", row=0, duration_ms=1200)
                continue

            api.exec_summary_ctx["selected_types"] = chosen

            # Precompute once
            chosen_label = ", ".join(s.title() for s in chosen)
            rows = [_finding_row(f) for f in (api.report_findings_content or [])]

            # Status line while generating
            try:
                max_y, max_x = stdscr.getmaxyx()
                stdscr.move(0, 0); stdscr.clrtoeol()
                stdscr.addnstr(0, 2, f"Generating executive summary for: {chosen_label}…", max_x - 4, curses.A_BOLD)
                stdscr.refresh()
            except curses.error:
                pass

            # Generate (this should also do post-process & stash inside the function)
            sections = api.generate_executive_summary(
                template_path="templates/execsummary.yml",
                use_llama=api.use_llama,
                use_grammarly=api.use_grammarly,
                selected_types=chosen,
                extra_ctx={
                    "FINDINGS_ROWS": "\n".join(rows),
                    "CLIENT_NAME": api.client_name or "",
                    "REPORT_NAME": api.report_name or "",
                }
            )

            # Build visual diffs once (after sections are saved)
            api.generate_visual_reportdiff()

            # Success toast
            _toast(
                stdscr,
                f"Executive summary generated ({len(sections)} sections). Press d to view diff or u to review.",
                row=0,
                duration_ms=1400
            )



def import_llm_suggestions(api, stdscr, max_x, max_y):
    """Use precomputed visual_diff_chunks to review and apply suggestions."""
    if not api.suggestedfixes_from_llm:
        print("Error: No LLM suggestions loaded.")
        return

    curses.curs_set(0)  # Hide cursor

    # === EXECUTIVE SUMMARY FIRST ===
    stdscr.clear()
    stdscr.addstr(0, 2, "Reviewing Executive Summary Sections (Update Mode)")
    stdscr.refresh()
    time.sleep(1)

    exec_summary_fields = deepcopy(api.report_content.get("exec_summary", {}).get("custom_fields", []))
    updated_exec_summary_fields = deepcopy(
        (api.suggestedfixes_from_llm.get("executive_summary_custom_fields", {}) or {}).get("custom_fields") or []
    )
    for idx, (field, updated_field) in enumerate(zip(exec_summary_fields, updated_exec_summary_fields)):

        field_id = field.get("id")
        label = field.get("label", f"Executive Summary Section {idx}")

        original_text = field.get("text", "")
        suggested_text = updated_field.get("text", "")

        # Skip if identical
        if (original_text or "").strip() == (suggested_text or "").strip():
            continue

        decision, staged_text, hunks = review_section_changes_tui(
            stdscr, section_label=label, original_text=original_text, suggested_text=suggested_text
        )

        if decision == "quit":
            return
        elif decision in ("advance", "none"):
            # nothing to apply here; move to next section
            continue
        if decision == "apply":
            field["text"] = updated_field["text"]
            api.update_executive_summary(field_id, staged_text)

    # === THEN FINDINGS ===
    stdscr.clear()
    stdscr.addstr(0, 2, "Reviewing Findings Sections (Update Mode)")
    stdscr.refresh()
    time.sleep(1)

    findings = deepcopy(api.report_findings_content or [])
    updated_findings = deepcopy((api.suggestedfixes_from_llm or {}).get("findings") or [])

    fields_to_review = ["title", "description", "recommendations", "guidance", "reproduction_steps"]

    for idx, (finding, updated_finding) in enumerate(zip(findings, updated_findings)):
        finding_id = finding.get("flaw_id")

        for field_name in fields_to_review:
            if field_name in ["guidance", "reproduction_steps"]:
                original_text = finding.get("fields", {}).get(field_name, {}).get("value", "")
                suggested_text = updated_finding.get("fields", {}).get(field_name, {}).get("value", "")
                label = f"Finding {idx} – {field_name.replace('_',' ').title()}"
            else:
                original_text = finding.get(field_name, "")
                suggested_text = updated_finding.get(field_name, "")
                label = f"Finding {idx} – {field_name.title()}"

            if not (original_text or suggested_text):
                continue
            if (original_text or "").strip() == (suggested_text or "").strip():
                continue

            decision, staged_text, hunks = review_section_changes_tui(
                stdscr, section_label=label, original_text=original_text, suggested_text=suggested_text
            )
            if decision == "quit":
                return
            elif decision in ("advance", "none"):
                # nothing to apply here; move to next section
                continue
            elif decision == "apply":
                # Route the staged text to the right field(s), then PUT via your existing API
                kwargs = {
                    "updated_title": finding.get("title"),
                    "updated_description": finding.get("description"),
                    "updated_recommendations": finding.get("recommendations"),
                    "updated_guidance": finding.get("fields", {}).get("guidance", {}).get("value", None),
                    "updated_reproduction_steps": finding.get("fields", {}).get("reproduction_steps", {}).get("value", None),
                }

                if field_name in ["guidance", "reproduction_steps"]:
                    # mutate the local copy
                    finding.setdefault("fields", {}).setdefault(field_name, {})["value"] = staged_text
                    kwargs[f"updated_{field_name}"] = staged_text
                else:
                    finding[field_name] = staged_text
                    kwargs[f"updated_{field_name}"] = staged_text

                api.update_finding(finding_id, **kwargs)

    stdscr.clear()
    stdscr.addstr(0, 2, "Report review complete!")
    stdscr.refresh()
    time.sleep(2)


def open_vi_to_edit(text):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(text.encode('utf-8'))
        temp_file.close()

        subprocess.run(['vi', temp_file.name])

        with open(temp_file.name, 'r', encoding='utf-8') as f:
            edited_content = f.read()

        os.remove(temp_file.name)

    return edited_content

def paginate_text(text, max_lines=300):
    """Splits long text into pages of a fixed number of lines."""
    lines = text.split("\n")
    return [lines[i:i+max_lines] for i in range(0, len(lines), max_lines)]

def generate_paginated_text(all_text, max_lines=300):
    paginated_text = [paginate_text(text, max_lines=max_lines) for text in all_text]
    paginated_text = [page for sublist in paginated_text for page in sublist]
    return paginated_text

def review_section_changes_tui(stdscr, section_label: str, original_text: str, suggested_text: str) -> tuple[str, str, list[Hunk]]:
    """
    Pure-TUI per-change review inside one screen.
    Returns: (decision, staged_text, hunks)
      decision: 'apply' (Enter) | 'quit' (q) | 'none' (no changes)
      staged_text: final text after accepted/edited hunks
      hunks: list with final per-hunk states (accepted/edited/rejected)
    """
    # helper inside review_section_changes_tui()
    def _counts():
        acc = sum(1 for x in hunks if x.state == "accepted")
        edt = sum(1 for x in hunks if x.state == "edited")
        rej = sum(1 for x in hunks if x.state == "rejected")
        pen = sum(1 for x in hunks if x.state == "pending")
        return acc, edt, rej, pen

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # additions
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # deletions
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)    # title
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # section
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)    # selection overlay (reverse-ish)


    focused_mode = True   # True = focused hunk view; False = full diff view
    scroll_offset = 0     # keep local scroll like your diff viewer

    # Build hunks once
    hunks = build_sentence_hunks(original_text, suggested_text)
    if not hunks:
        # Nothing to do
        return ("none", original_text, [])

    selected = 0
    staged_text = original_text

    def _render():
        nonlocal scroll_offset, focused_mode, selected
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()

        # --- Row 0: MENU (keys) ---
        header = "(a=accept change, s=skip change, e=edit+accept, ↑/↓ scroll, PgUp/PgDn page, n/p next/previous change, v=view full diff, f=clean diff, Enter=apply, q=quit)"
        try:
            stdscr.addnstr(0, 2, header, max_x - 4, curses.A_BOLD)
        except curses.error:
            pass

        # pick the chunk for the current view
        if focused_mode is True:
            chunk = build_visual_chunk_for_hunk(
                section_label, original_text, suggested_text, hunks[selected], ctx_lines=2
            )
        else:  # full
            chunk = _make_full_chunk(section_label, original_text, suggested_text)

        # render chunk starting on row 2 (so menu=0, title=1, content>=2)
        new_offset, _ = display_visual_diff_chunk_with_scroll(
            stdscr, chunk, scroll_offset, start_row=2, left_pad=2
        )
        scroll_offset = new_offset

        # footer on last row as before...
        # counts (locals inside _render)
        accepted = sum(1 for h in hunks if h.state == "accepted")
        edited   = sum(1 for h in hunks if h.state == "edited")
        rejected = sum(1 for h in hunks if h.state == "rejected")
        pending  = sum(1 for h in hunks if h.state == "pending")

        footer = f"[accepted {accepted} | edited {edited} | rejected {rejected} | pending {pending}]"

        # draw on the last row
        try:
            stdscr.addnstr(max_y - 1, 2, footer, max_x - 4, curses.A_DIM)
        except curses.error:
            pass

        stdscr.refresh()

    _render()

    while True:
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')):
            ###selected = (selected - 1) % len(hunks)
            # scroll up one line
            scroll_offset = max(0, scroll_offset - 1)
            _render()
        elif key in (curses.KEY_DOWN, ord('j')):
            ###selected = (selected + 1) % len(hunks)
            # scroll down one line
            scroll_offset += 1
            _render()
        elif key == ord('a'):
            hunks[selected].state = "accepted"
            staged_text = apply_hunks_to_text(original_text, hunks)
            _render()
        elif key == ord('s'):
            hunks[selected].state = "rejected"
            staged_text = apply_hunks_to_text(original_text, hunks)
            _render()
        elif key == ord('e'):
            # Edit suggested text in vi, then mark edited
            edited = open_vi_to_edit(hunks[selected].suggested_text or "")
            curses.cbreak(); curses.noecho(); stdscr.keypad(True); curses.curs_set(0)  # re-init after vi
            hunks[selected].suggested_text = edited
            hunks[selected].state = "edited"
            staged_text = apply_hunks_to_text(original_text, hunks)
            _render()
        elif key in (10, 13):  # Enter
            return ("apply", staged_text, hunks)
        elif key == ord('q'):
            return ("quit", staged_text, hunks)
        elif key == ord('v'):
            focused_mode = False
            scroll_offset = 0
            _render()
        elif key == ord('f'):
            focused_mode = True
            scroll_offset = 0
            _render()
        elif key in (curses.KEY_NPAGE, ord(' ')):   # page down
            #scroll_offset += (max_y - 5)
            max_y, _ = stdscr.getmaxyx()
            scroll_offset += (max_y - 4)
            _render()
        elif key == curses.KEY_PPAGE:               # page up
            ##scroll_offset = max(0, scroll_offset - (max_y - 5))
            max_y, _ = stdscr.getmaxyx()
            scroll_offset = max(0, scroll_offset - (max_y - 4))
            _render()
        elif key == ord('n') and focused_mode is True:
            selected = (selected + 1) % len(hunks)
            scroll_offset = 0
            _render()
        elif key == ord('p') and focused_mode is True:
            selected = (selected - 1) % len(hunks)
            scroll_offset = 0
            _render()


        acc, edt, rej, pen = _counts()
        if pen == 0 and (acc + edt) == 0:
            # all decided, but nothing to apply -> auto-advance to next section
            return ("advance", original_text, hunks)


def display_visual_diff_mode(stdscr, visual_diff):
    """Display the word-based, colorized diff in a scrollable curses interface."""
    stdscr.clear()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Additions
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)  # Deletions
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Titles
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Sections
    curses.curs_set(0)  # Hide cursor

    max_y, max_x = stdscr.getmaxyx()
    pad_height = max(len(visual_diff) * 2, max_y) + 10  # Ensure enough space
    pad = curses.newpad(pad_height, max_x)
    y_offset = 0  # Scroll position
    line_num = 0  # Track line position

    # Render text into the pad
    for diff_type, text in visual_diff:
        # Wrap text properly to fit screen width
        wrapped_lines = textwrap.wrap(text, width=max_x - 4)

        # Apply color formatting
        color = curses.color_pair(1) if diff_type == "add" else \
                curses.color_pair(2) if diff_type == "remove" else \
                curses.color_pair(3) if diff_type == "title" else \
                curses.color_pair(4) if diff_type == "section" else 0

        for line in wrapped_lines:
            if line_num < pad_height - 1:
                pad.addstr(line_num, 2, line, color)
                line_num += 1  # Move to next line normally

    while True:
        stdscr.refresh()  # Keep background stable
        stdscr.addstr(0, 0, "DIFF View (q-back, up arrow-scroll up, down arrow-scroll down)")

        # Refresh pad display with correct offset
        pad.refresh(y_offset, 0, 1, 0, max_y - 1, max_x - 1)

        key = stdscr.getch()
        if key == ord('q'):  # Quit the diff view
            break
        elif key == curses.KEY_DOWN and y_offset < max(pad_height - max_y, 0):
            y_offset += 1  # Scroll down (prevent over-scrolling)
        elif key == curses.KEY_UP and y_offset > 0:
            y_offset -= 1  # Scroll up


# For generation of executive summary.  The user will select one or more.
TEST_TYPES = [
    ("external","External"),
    ("internal","Internal"),
    ("webapp","Web App"),
    ("mobile","Mobile"),
    ("cloud","Cloud / Config"),
    ("social","Social / Phishing"),
]

def pick_test_types(stdscr, preselected=None):
    pre = set(preselected or [])
    idx, scroll = 0, 0
    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        stdscr.addnstr(0, 2, "Select test types: Space=toggle, Enter=OK, q=cancel", max_x-4, curses.A_BOLD)
        view = TEST_TYPES
        h = max_y - 3
        scroll = max(0, min(scroll, max(0, len(view)-h)))
        for i in range(h):
            j = scroll + i
            if j >= len(view): break
            key, label = view[j]
            sel = "[x]" if key in pre else "[ ]"
            attr = curses.A_REVERSE if j == idx else 0
            stdscr.addnstr(2+i, 4, f"{sel} {label}", max_x-8, attr)
        stdscr.refresh()
        k = stdscr.getch()
        if k in (ord('q'), 27): return None
        elif k in (curses.KEY_UP, ord('k')): idx = max(0, idx-1);  scroll = min(scroll, idx)
        elif k in (curses.KEY_DOWN, ord('j')): idx = min(len(view)-1, idx+1); scroll = max(scroll, idx-h+1)
        elif k in (ord(' '),):
            key = view[idx][0]
            if key in pre: pre.remove(key)
            else: pre.add(key)
        elif k in (10, 13):  # Enter
            return list(pre)


def show_exec_summary_fitness_report(stdscr, api) -> None:
    import curses
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()

    # Get original exec summary (your getter returns list[str] for pagination)
    exec_pages = api.get_local_exec_summary(original=True) or []
    exec_text = _ef_concat_pages(exec_pages)
    findings = api.report_findings_content or []

    result = assess_exec_summary_executive_fitness(exec_text, findings)

    header = f"[Exec Fitness]  Score: {result['score']}/100   Severity aligned: {'YES' if result['severity_aligned'] else 'NO'} (summary: {result['summary_max_sev']}, findings: {result['findings_max_sev']})"
    lines = []
    sec = result["sections"]

    def add_block(title, kv_pairs):
        lines.append(title)
        for k,v in kv_pairs:
            lines.append(f"  • {k}: {v}")
        lines.append("")

    add_block("Synthesis vs Enumeration",
              [("score", sec["synthesis"]["score"]),
               ("synthesis_hits", sec["synthesis"]["synthesis_hits"]),
               ("bullet_lines", sec["synthesis"]["bullet_lines"]),
               ("enum_matches", ", ".join(sec["synthesis"]["enum_matches"]) or "none")])

    add_block("Theme Coverage",
              [("score", sec["themes"]["score"]),
               ("top_themes", ", ".join(sec["themes"]["top_themes"]) or "none"),
               ("mentioned", ", ".join(sec["themes"]["mentioned"]) or "none"),
               ("missing", ", ".join(sec["themes"]["missing"]) or "none")])

    add_block("Business Impact",
              [("score", sec["impact"]["score"]),
               ("impact_terms_found", ", ".join(sec["impact"]["impact_terms_found"]) or "none")])

    add_block("Recommendations",
              [("score", sec["recommendations"]["score"]),
               ("action_verbs", ", ".join(sec["recommendations"]["action_verbs"]) or "none"),
               ("priority_cues", ", ".join(sec["recommendations"]["priority_cues"]) or "none"),
               ("hedges", ", ".join(sec["recommendations"]["hedges"]) or "none")])

    add_block("Readability",
              [("score", sec["readability"]["score"]),
               ("avg_sentence_len", sec["readability"]["avg_sentence_len"]),
               ("acronym_density", sec["readability"]["acronym_density"]),
               ("acronyms", ", ".join(sec["readability"]["acronyms"]) or "none")])

    # Scrollable render
    start_row, left_pad, scroll = 1, 2, 0
    while True:
        stdscr.erase()
        try:
            stdscr.addnstr(0, 2, header, max_x-4, curses.A_BOLD)
        except curses.error: pass

        max_y, max_x = stdscr.getmaxyx()
        content_h = max_y - start_row - 1
        max_off = max(0, len(lines) - content_h)
        scroll = max(0, min(scroll, max_off))

        for i in range(content_h):
            j = scroll + i
            if j >= len(lines): break
            try:
                stdscr.addnstr(start_row + i, left_pad, str(lines[j]), max_x-4)
            except curses.error: pass

        try:
            footer = f"[{scroll+1}-{min(scroll+content_h, len(lines))}/{len(lines)}]  ↑/↓ scroll · PgUp/PgDn page · q back"
            stdscr.addnstr(max_y-1, 2, footer, max_x-4, curses.A_DIM)
        except curses.error: pass

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord('q'), 27): return
        elif key in (curses.KEY_DOWN, ord('j')): scroll += 1
        elif key in (curses.KEY_UP, ord('k')): scroll -= 1
        elif key in (curses.KEY_NPAGE, ord(' ')): scroll += max(1, content_h-1)
        elif key == curses.KEY_PPAGE: scroll -= max(1, content_h-1)


def show_exec_summary_consistency_report(stdscr, api) -> None:
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()

    header = "[C] Exec Summary Consistency — ↑/↓ scroll, q quit"
    try:
        stdscr.addnstr(0, 2, header, max_x - 4, curses.A_BOLD)
    except curses.error:
        pass

    exec_pages = api.get_local_exec_summary(original=True) or []
    exec_concat = _concat_exec_summary_from_pages(exec_pages)

    findings = api.report_findings_content or []

    # NEW: map ids -> titles (robust to different ID fields)
    id_to_title = {}
    for i, f in enumerate(findings):
        key = str(
            f.get("id")
            or f.get("finding_id")
            or f.get("flaw_id")
            or f.get("title")
            or f"idx:{i}"
        )
        title = (f.get("title") or f"Untitled Finding {i+1}").strip()
        id_to_title[key] = title

    result = assess_exec_summary_consistency(exec_concat, findings)

    lines = []
    lines.append(f"Confidence: {result.confidence}/100")
    lines.append(
        f"Severity aligned: {'YES' if result.severity_alignment_ok else 'NO'} "
        f"(summary: {result.summary_max_sev}, findings: {result.findings_max_sev})"
    )
    lines.append("")

    if result.contradictions:
        lines.append("Contradictions:")
        for c in result.contradictions:
            lines.append(f"  - {c}")
        lines.append("")

    # --- Covered section (now with titles)
    lines.append(f"Covered findings: {len(result.covered_ids)} / {len(findings)}")
    for fid in result.covered_ids[:20]:
        cov = result.coverage.get(fid)
        sc = cov.score if cov else 0.0
        title = id_to_title.get(fid, fid)
        ev = f"  {cov.evidence}" if (cov and cov.evidence) else ""
        lines.append(f"  ✓ {title}  (score {sc}){ev}")

    # --- Uncovered section (now with titles)
    if result.uncovered_ids:
        lines.append("")
        lines.append(f"Uncovered findings: {len(result.uncovered_ids)}")
        for fid in result.uncovered_ids[:50]:
            cov = result.coverage.get(fid)
            sc = cov.score if cov else 0.0
            title = id_to_title.get(fid, fid)
            lines.append(f"  ✗ {title}  (score {sc})")


    if result.theme_gaps:
        lines.append("")
        lines.append("Theme gaps:")
        for g in result.theme_gaps[:10]:
            lines.append(f"  - {g['theme']}: {', '.join(g['example_findings'])}")

    # render scrollable
    start_row = 1
    left_pad = 2
    scroll = 0
    while True:
        stdscr.erase()
        try:
            stdscr.addnstr(0, 2, header, max_x - 4, curses.A_BOLD)
        except curses.error:
            pass

        max_y, max_x = stdscr.getmaxyx()
        content_h = max_y - start_row - 1
        max_off = max(0, len(lines) - content_h)
        scroll = max(0, min(scroll, max_off))

        for i in range(content_h):
            j = scroll + i
            if j >= len(lines): break
            try:
                stdscr.addnstr(start_row + i, left_pad, lines[j], max_x - 4)
            except curses.error:
                pass

        try:
            footer = f"[{scroll+1}-{min(scroll+content_h, len(lines))}/{len(lines)}]  q quit"
            stdscr.addnstr(max_y - 1, 2, footer, max_x - 4, curses.A_DIM)
        except curses.error:
            pass

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord('q'), 27):
            return
        elif key in (curses.KEY_DOWN, ord('j')):
            scroll += 1
        elif key in (curses.KEY_UP, ord('k')):
            scroll -= 1
        elif key in (curses.KEY_NPAGE, ord(' ')):
            scroll += max(1, content_h - 1)
        elif key == curses.KEY_PPAGE:
            scroll -= max(1, content_h - 1)


def build_sentence_hunks(original: str, suggested: str) -> list[Hunk]:
    """Create sentence-level hunks between original and suggested."""
    orig_sents = _split_sentences_for_diff(original)
    sugg_sents = _split_sentences_for_diff(suggested)

    sm = SequenceMatcher(a=orig_sents, b=sugg_sents, autojunk=False)
    hunks: list[Hunk] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            left_cnt = i2 - i1
            right_cnt = j2 - j1

            if left_cnt == right_cnt:
                # 1:1 (or n:n) — keep per-sentence replace hunks
                for k, (o, s) in enumerate(zip(orig_sents[i1:i2], sugg_sents[j1:j2])):
                    hid = f"rep@{i1+k}:{hash(o) & 0xffff:x}"
                    hunks.append(Hunk(hid, "replace", i1 + k, o, s))
            else:
                # UNEVEN replace (e.g., 2->1 or 1->2 or 3->1 …)
                # Emit a SINGLE block hunk instead of zipping + leftovers.
                orig_block = " ".join(orig_sents[i1:i2])
                sugg_block = " ".join(sugg_sents[j1:j2])
                hid = f"repblk@{i1}-{i2}:{hash(orig_block) & 0xffff:x}"
                hunks.append(Hunk(hid, "replace", i1, orig_block, sugg_block))
        elif tag == "delete":
            for k, o in enumerate(orig_sents[i1:i2]):
                hid = f"del@{i1+k}:{hash(o) & 0xffff:x}"
                hunks.append(Hunk(hid, "delete", i1+k, o, ""))
        elif tag == "insert":
            anchor = i1 - 1
            for k, s in enumerate(sugg_sents[j1:j2]):
                hid = f"ins@{anchor+1}+{k}:{hash(s) & 0xffff:x}"
                hunks.append(Hunk(hid, "insert", anchor, "", s))

    # Keep deterministic order by original sentence position, then kind
    def _sort_key(h: Hunk):
        kind_rank = {"delete": 0, "replace": 1, "insert": 2}
        return (h.sent_idx_orig, kind_rank.get(h.change_kind, 9))
    hunks.sort(key=_sort_key)
    return hunks

def display_visual_diff_chunk_with_scroll(
    stdscr,
    visual_diff_chunk: list[tuple[str, str]],
    scroll_offset: int,
    *,
    start_row: int = 2,
    left_pad: int = 2,
) -> tuple[int, int]:
    """
    Render a diff 'chunk' with wrapping and vertical scrolling.
    - visual_diff_chunk: list of (kind, text); kind in {"section","add","remove","normal"}
    - scroll_offset: first *wrapped* line index to render
    - start_row: top row for content (header sits above)
    - left_pad: left margin spaces

    Returns (clamped_scroll_offset, total_wrapped_lines)
    """
    max_y, max_x = stdscr.getmaxyx()
    content_height = max(0, max_y - start_row - 1)  # leave one row for footer/status

    # Build wrapped lines with color metadata
    wrapped: list[tuple[int, str]] = []  # (color_attr, text)

    # Color pairs expected (init these elsewhere once):
    # 1: green additions, 2: red deletions, 3: cyan title, 4: yellow section
    add_attr    = curses.color_pair(1) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD
    remove_attr = curses.color_pair(2) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD
    title_attr  = curses.color_pair(3) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD
    sect_attr   = curses.color_pair(4) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD
    norm_attr   = curses.A_NORMAL

    wrap_width = max(10, max_x - (left_pad + 2))  # gutter + pad

    for kind, text in visual_diff_chunk:
        # prefix + choose color
        if kind == "section":
            prefix = ""
            attr = sect_attr
        elif kind == "add":
            prefix = "+ "
            attr = add_attr
        elif kind == "remove":
            prefix = "- "
            attr = remove_attr
        else:
            prefix = ""
            attr = norm_attr

        # Split into visual paragraphs first (keep structure)
        paragraphs = (text or "").split("\n")
        for para in paragraphs:
            para = para.rstrip()
            if not para:
                # preserve blank line separation
                wrapped.append((norm_attr, ""))  # a blank visual line
                continue

            # Wrap with prefix on first line, indent following lines to align nicely
            first = True
            for seg in textwrap.wrap(para, width=wrap_width) or [""]:
                if first and prefix:
                    line = prefix + seg
                else:
                    line = ("  " if prefix else "") + seg  # indent continuation lines if we had a +/- prefix
                wrapped.append((attr, line))
                first = False

    total_wrapped = len(wrapped)
    if total_wrapped == 0:
        return (0, 0)

    # Clamp scroll_offset
    max_offset = max(0, total_wrapped - content_height)
    scroll_offset = max(0, min(scroll_offset, max_offset))

    # Determine slice to render
    start = scroll_offset
    end = min(total_wrapped, start + content_height)

    # Draw visible lines
    row = start_row
    for i in range(start, end):
        attr, line = wrapped[i]
        # left gutter indicator if the line is an add/remove
        gutter = "  "
        if attr == add_attr:
            gutter = "▶ "
        elif attr == remove_attr:
            gutter = "◀ "
        # paint
        try:
            stdscr.addstr(row, left_pad, gutter, curses.A_DIM)
            stdscr.addnstr(row, left_pad + len(gutter), line, max(0, max_x - (left_pad + len(gutter))), attr)
        except curses.error:
            pass
        row += 1

    return (scroll_offset, total_wrapped)

def build_visual_chunk_for_hunk(
    section_label: str,
    original_text: str,
    suggested_text: str,
    hunk: Hunk,
    ctx_lines: int = 2
) -> list[tuple[str, str]]:
    """
    Build a scrollable, structured chunk for the selected hunk with:
      • original-side context lines
      • focused core (remove/add)
      • suggested-side context lines
    Uses HTML-aware line splitting so context matches what full diff shows.
    """
    chunk: list[tuple[str, str]] = [("section", f"{section_label}:")]

    # ---- Structure-aware lines on both sides
    old_lines = _split_lines_htmlaware(original_text)
    new_lines = _split_lines_htmlaware(suggested_text)

    # Sentences (for anchoring); reuse your sentence splitter
    orig_sents = _split_sentences_for_diff(original_text)
    sugg_sents = _split_sentences_for_diff(suggested_text)

    # Helper: map a char position in full text to the line index inside a prepared line list
    def _charpos_to_lineindex(full_text: str, lines: list[str], charpos: int) -> int:
        acc = 0
        for i, L in enumerate(lines):
            # +1 accounts for the '\n' we introduced when splitting
            if acc <= charpos < acc + len(L) + 1:
                return i
            acc += len(L) + 1
        return max(0, len(lines) - 1)

    # ---- ORIGINAL-SIDE window around the anchor sentence
    # For inserts we still show some original context (may be nearby, may be blank if none).
    sent_idx = hunk.sent_idx_orig if hunk.sent_idx_orig is not None else 0
    sent_idx = max(0, min(sent_idx, len(orig_sents) - 1)) if orig_sents else 0
    anchor_sentence = orig_sents[sent_idx] if orig_sents else ""

    old_start = 0
    old_end = min(len(old_lines), 1 + ctx_lines)  # default small window if we can't locate
    if anchor_sentence:
        pos_old = original_text.find(anchor_sentence)
        if pos_old >= 0:
            line_idx_old = _charpos_to_lineindex(original_text, old_lines, pos_old)
            old_start = max(0, line_idx_old - ctx_lines)
            old_end = min(len(old_lines), line_idx_old + ctx_lines + 1)

    # ---- SUGGESTED-SIDE window centered near the new content
    # Pick the first suggested sentence in the hunk; fallback to the hunk's suggested_text.
    sugg_anchor_sentence = ""
    if hunk.change_kind in ("replace", "insert"):
        # Try to pick the sentence at the same index; clamp
        if sugg_sents:
            sugg_anchor_sentence = sugg_sents[min(sent_idx, len(sugg_sents) - 1)]
        else:
            sugg_anchor_sentence = hunk.suggested_text
    elif hunk.change_kind == "delete":
        # No new text; still show a small window starting where the deletion occurs
        sugg_anchor_sentence = ""

    new_start = 0
    new_end = 0
    if sugg_anchor_sentence:
        pos_new = suggested_text.find(sugg_anchor_sentence)
        if pos_new >= 0:
            line_idx_new = _charpos_to_lineindex(suggested_text, new_lines, pos_new)
            new_start = max(0, line_idx_new - ctx_lines)
            new_end = min(len(new_lines), line_idx_new + ctx_lines + 1)

    # ---- Emit original-side context
    for i in range(old_start, old_end):
        # Keep blank lines as actual blanks in the viewer
        if old_lines[i] == "":
            chunk.append(("normal", ""))
        else:
            chunk.append(("normal", old_lines[i]))

    # ---- Focused core (trim prefix/suffix for replaces)
    if hunk.change_kind == "replace":
        old_core, new_core = _focus_changed_span(hunk.original_text, hunk.suggested_text)
        if old_core.strip():
            chunk.append(("remove", old_core))
        if new_core.strip():
            chunk.append(("add", new_core))
    elif hunk.change_kind == "delete":
        if (hunk.original_text or "").strip():
            chunk.append(("remove", hunk.original_text))
    elif hunk.change_kind == "insert":
        if (hunk.suggested_text or "").strip():
            chunk.append(("add", hunk.suggested_text))

    # ---- Visual spacer if we will also show suggested-side context
    if new_end > new_start and (old_end > old_start):
        chunk.append(("normal", ""))

    # ---- Emit suggested-side context (guarantees new tokens appear in the blurb)
    for i in range(new_start, new_end):
        if new_lines[i] == "":
            chunk.append(("normal", ""))
        else:
            chunk.append(("normal", new_lines[i]))

    return chunk


def apply_hunks_to_text(original: str, hunks: list[Hunk]) -> str:
    """Apply accepted/edited hunks to original text, sentence-level."""
    sents = _split_sentences_for_diff(original)
    # We'll build a working list we can insert/delete/replace
    work = list(sents)

    # To avoid index drift, apply in increasing anchor order while tracking offset adjustments
    offset = 0
    for h in hunks:
        if h.state not in ("accepted", "edited"):
            continue
        idx = h.sent_idx_orig + offset
        if h.change_kind == "replace":
            # Support both single-sentence and block (multi-sentence) replaces.
            orig_block_sents = _split_sentences_for_diff(h.original_text)
            sugg_block_sents = _split_sentences_for_diff(h.suggested_text)

            if len(orig_block_sents) <= 1 and len(sugg_block_sents) <= 1:
                # simple 1 -> 1
                if 0 <= idx < len(work):
                    work[idx] = h.suggested_text
                else:
                    # if index fell off due to prior edits, clamp/append
                    if idx < 0:
                        work.insert(0, h.suggested_text)
                        offset += 1
                    else:
                        work.append(h.suggested_text)
                        offset += 1
            else:
                # block replace: replace the slice [idx : idx + len(orig_block_sents)]
                start = max(0, min(idx, len(work)))
                end = start + len(orig_block_sents)
                end = max(start, min(end, len(work)))
                work[start:end] = sugg_block_sents
                offset += (len(sugg_block_sents) - (end - start))
        elif h.change_kind == "delete":
            if 0 <= idx < len(work):
                del work[idx]
                offset -= 1
        elif h.change_kind == "insert":
            # insert AFTER the anchor index
            insert_at = idx + 1
            if insert_at < 0:
                insert_at = 0
            if insert_at > len(work):
                work.append(h.suggested_text)
            else:
                work.insert(insert_at, h.suggested_text)
                offset += 1

    # Re-join sentences with a single space (matches your original splitter expectation)
    return " ".join(work).strip()
