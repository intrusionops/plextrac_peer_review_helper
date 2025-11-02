import curses
import textwrap
import requests
import argparse
import json
from datetime import datetime
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
        _english_test_types,
        merge_exec_fields_for_views,
        assess_exec_summary_consistency,
        is_executive_summary_unchanged,
        is_finding_title_unchanged,
        is_finding_description_unchanged,
        is_finding_recommendations_unchanged,
        is_finding_guidance_unchanged,
        is_finding_reproduction_steps_unchanged
)

from user_interface import (
        soft_format_html_for_terminal,
        interactive_text_viewer,
        import_llm_suggestions,
        open_vi_to_edit,
        paginate_text,
        generate_paginated_text,
        review_section_changes_tui,
        display_visual_diff_mode,
        TEST_TYPES,
        pick_test_types,
        show_exec_summary_fitness_report,
        show_exec_summary_consistency_report,
        build_sentence_hunks,
        display_visual_diff_chunk_with_scroll,
        build_visual_chunk_for_hunk,
        apply_hunks_to_text,
        Hunk
)


os.system("stty -ixon")  # Disable CTRL+S freezing in the terminal

class PlexTracAPI:
    def __init__(self, server_fqdn, username, password, ollama_remote_url=None, tgi_remote_url=None, 
                use_llama=True, use_grammarly=False, use_html_wrapping=False):
        self.base_url = f"https://{server_fqdn}/api/v1"
        self.username = username
        self.password = password
        self.token = None # Authentication response jwt from plextrac
        self.session = requests.Session()  # Create a session object
        self.clients = None # JSON - returned from call to plexapi GET /api/v1/client/list
        
        self.client = None # JSON - returned from call to plexapi GET /api/v1/client/list for our target client
        self.client_id = None # String - Client whose report we're interested in
        self.client_name=None
        self.report_name=None

        # Original report
        self.reports=None
        self.report_info = None # JSON - return from call to plexapi GET /api/v1/client/{client_id}/reports for our target report
        self.report_id = None # String
        self.report_content = None # JSON - from call to /api/v1/client/{client_id}/report/{report}
        self.report_findings_list = None # JSON - returned from GET /api/v1/client/{client_id/report/{report_id}/flaws
        self.report_findings_content = [] # JSON [] - returned from getting each finding

        # Whether to send the html objects or extract them before sending to llm
        self.use_html_wrapping=use_html_wrapping

        # LLM Suggested fixes for report
        self.ollama_remote_url = ollama_remote_url
        self.retrieved_suggestedfixes_from_llm = False
        self.suggestedfixes_from_llm = {"executive_summary_custom_fields": None, "findings": None}
        self.visual_diff_generated=False
        self.visual_diff=None # this will hold the comparison result between original text and llm suggested text
        self.visual_diff_chunks=[] # This is for displaying each diffed finding or exec summary section during updating

        self.exec_summary_ctx = {
            "selected_types": [],   # What kind of test is this - external, int, webapp... - user makes this choice when generating/regenerating exec summary
        }

        # LLM Suggested fixes via Grammarly Model (TGI Interface or locally)
        self.tgi_remote_url=tgi_remote_url
        self.copy_editor = CopyEditor()
        if use_grammarly:
            self.copy_editor.load_grammarly_model(tgi_remote_url=tgi_remote_url, model_name="grammarly/coedit-xl")
        if use_llama:
            self.copy_editor.llama=LlamaModel(ollama_remote_url=ollama_remote_url)

        self.use_llama=use_llama
        self.use_grammarly=use_grammarly

    # Authentication Functions
    # -------------------------------------------------------------------------------------------------------

    def authentication_required_decorator(func):
        """Decorator to check if authentication (self.token) is available."""
        def wrapper(self, *args, **kwargs):
            if not self.token:
                print("Authentication required.")
                return None
            return func(self, *args, **kwargs)
        return wrapper

    def authenticate(self):
        """Authenticate and retrieve JWT token."""
        auth_url = f"{self.base_url}/authenticate"
        response = self.session.post(auth_url, json={"username": self.username, "password": self.password})
        if response.status_code == 200:
            self.token = response.json().get('token')
            print("Authentication successful.")
            self.start_token_refresh()
        else:
            print(f"Authentication failed: {response.status_code} - {response.text}")
            exit(1)

    def refresh_token(self):
        """Refresh the JWT token using the refresh endpoint."""
        refresh_url = f"{self.base_url}/token/refresh"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        response = self.session.put(refresh_url, headers=headers)
        
        if response.status_code == 200:
            # Extract new token and cookie from the response
            self.token = response.json().get('token')
        else:
            print(f"Failed to refresh token: {response.status_code} - {response.text}")
            exit(1)

    def start_token_refresh(self):
        """Start a background thread to refresh the token every 10 minutes."""
        if self.token is None:
            print(" Cannot start token refresh—user is not authenticated.")
            return

        def refresh_loop():
            while self.token is not None:  # Run only if authenticated
                time.sleep(600)  # Wait 10 minutes
                self.refresh_token()

        threading.Thread(target=refresh_loop, daemon=True).start()
        print("Token refresh loop started.")
    # -------------------------------------------------------------------------------------------------------



    # Data request functions
    # -------------------------------------------------------------------------------------------------------

    @authentication_required_decorator
    def make_request_authenticated(self, method, url, headers=None, params=None, data=None, json=None):
        """Helper function to make an API request with automatic token refresh if expired."""

        headers = headers or {}
        headers["Authorization"] = f"Bearer {self.token}"

        response = self.session.request(method, url, headers=headers, params=params, data=data, json=json)

        # If token is expired (401 status), refresh and retry the request
        if response.status_code == 401:
            print("Session expired, refreshing token...")
            self.refresh_token()
            headers["Authorization"] = f"Bearer {self.token}"  # Update the headers with new token
            response = self.session.request(method, url, headers=headers, params=params, data=data)

        return response

    @authentication_required_decorator
    def get_client(self, client_name):
        """retrieves a client json blob and returns it as well as storing it in the class """
        # Fetch all clients
        clients_url = f"{self.base_url}/client/list"
        response = self.make_request_authenticated("GET", clients_url)
        if response.status_code == 200:
            self.clients = response.json()
            client = [ c for c in self.clients if c["data"][1].lower() == client_name.lower() ]
            if self.check_if_client_has_duplicate(client_name) is True:
                print("Multiple clients exist with the same name...please fix and then rerun")
                self.client_id=None
                self.client=None
                self.clients=None
            elif len(client)==1:
                self.client_id=client[0]['doc_id'][0]
                self.client=client[0]
                self.client_name=client_name
            return self.client
        else:
            print(f"Failed to retrieve clients: {response.status_code} - {response.text}")
            return None


    @authentication_required_decorator
    def get_report(self, client_name, report_name):
        """Check if a report exists for a given client by name."""
        # get client_id
        if self.client is None and self.get_client(client_name) is None:
            return None

        # Fetch reports for the client
        reports_url = f"{self.base_url}/client/{self.client_id}/reports"
        response = self.make_request_authenticated("GET", reports_url)

        target_report=[] # We make this a list to check if multiple reports with the same name exist.  So in the normal case 
                         # (e.g. single report with a given name) it should only be of length 1
        if response.status_code == 200:
            reports = response.json()
            self.reports = reports
            target_report=[ r for r in reports if r["data"][1].lower() == report_name.lower() ] 
            if self.check_if_report_has_duplicate(report_name) is True:
                print("Multiple reports exist with the same name...please fix and then rerun")
                self.report_info=None
            elif len(target_report)==1:
                self.report_info=target_report[0]
        if self.report_info is not None:
            self.report_id=self.report_info['data'][0]
            self.report_content=self.make_request_authenticated("GET", f"{self.base_url}/client/{self.client_id}/report/{self.report_id}")
            self.report_content=self.report_content.json()
            return self.report_content
        else:
            print(f"Failed to retrieve reports for client '{client_name}': {response.status_code} - {response.text}")
            return None
    
    @authentication_required_decorator
    def get_report_findings_list(self, client_name, report_name):
        """Get brief info on report findings and return as a list"""
        if self.report_content is None:
            report_response=self.get_report(client_name, report_name)
            if report_response is None:
                return None
        findings_response=self.make_request_authenticated("GET", f"{self.base_url}/client/{self.client_id}/report/{self.report_id}/flaws")
        if findings_response is not None:
            self.report_findings_list=findings_response.json()
            return self.report_findings_list
        else:
            print (f"Failed to retrieve findings list for '{client_name}': {response.status_code} - {response.text}")
            return None

    @authentication_required_decorator
    def get_report_findings_content(self, client_name, report_name):
        """Get all of the content for each report finding and return as a list"""
        if self.report_findings_list is None:
            report_response=self.get_report_findings_list(client_name, report_name)
            if report_response is None:
                return None

        for f in self.report_findings_list:
            findings_response=self.make_request_authenticated("GET", f"{self.base_url}/client/{self.client_id}/report/{self.report_id}/flaw/{f['data'][0]}")
            if findings_response is not None:
                self.report_findings_content.append( findings_response.json() )
            else:
                print (f"Failed to retrieve findings for '{client_name}': {response.status_code} - {response.text}")

        return self.report_findings_content

    @authentication_required_decorator
    def export_report(self, client_name, report_name, download_location="./"):
        """ Export a ptrac & docx copy of the report, NOTE: it's not necessary to fetch report first before exporting so fix later """
        exported_successfully=True

        if self.report_content is None:
            report_response=self.get_report(client_name, report_name)
            if report_response is None:
                return False
        ptrac=self.make_request_authenticated("GET", f"{self.base_url}/client/{self.client_id}/report/{self.report_id}/export/ptrac")
        docx=self.make_request_authenticated("GET", f"{self.base_url}/client/{self.client_id}/report/{self.report_id}/export/doc?includeEvidence=False")

        if ptrac is not None:
            ptrac=ptrac.json()
            with open(download_location+client_name+'_'+report_name+'.ptrac', 'w') as json_file:
                json.dump(ptrac, json_file)
        else:
            print ("Could not export ptrac file")
            exported_successfully=False

        if docx is not None:
            docx=docx.content
            with open(download_location+client_name+'_'+report_name+'.docx', 'wb') as bin_file:
                bin_file.write(docx)
        else:
            print ("Could not export docx file")
            exported_successfully=False

        return exported_successfully

    def check_if_report_has_duplicate(self, report_name):
        reports=[ r for r in self.reports if r["data"][1].lower() == report_name.lower() ]
        if len(reports)>1:
            return True
        return False

    def check_if_client_has_duplicate(self, client_name):
        clients=[ r for r in self.clients if r["data"][1].lower() == client_name.lower() ]
        if len(clients)>1:
            return True
        return False
    # --------------------------------------------------------------------------------------------


    # LLM Querying functions
    # --------------------------------------------------------------------------------------------
    def get_suggested_fixes_from_llm(self, use_llama=False, use_grammarly=True, prompts=None, use_html_wrapping=False):
        """Send executive summary and findings to LLM for modification suggestions (parallelized version)."""
        if not self.report_content or not self.report_findings_content:
            print("Error: Report content or findings are missing.")
            return

        # Prompts
        prompts = prompts or {
            "exec_summary": {
                "grammarly": "Make this text coherent and fix any grammar issues:",
                "llama": """You are a copy-editor. Improve clarity, flow, grammar, and professionalism while preserving meaning and any URLs/HTML tags.

    OUTPUT RULES:
    - Output ONLY the revised text.
    - Do NOT add any explanations, headings, labels, or lead-ins (e.g., “Here is the edited text:”).
    - Do NOT wrap the output in quotes or Markdown/code fences.
    - Keep all HTML tags that are present in the input unless they are clearly broken.

    Use these Strict instructions:
    - Do not add, remove, or rewrite sentences.
    - Do not invent any new information or make assumptions.
    - Only fix grammar, punctuation, and typos.
    - Remove roundabout phrases like: "during this engagement", "in order to", etc.
    - If a number is written as "five (5)", convert it to just "five". Do not change actual dates.
    - Do not add or remove any HTML tags, including <code>...</code> or <span>...</span>.
    - Keep company names capitalized as: PentestCompany
    - Replace verbose technical phrases with standard abbreviations, **only if they already exist in the text**.

    Return only the edited version of the input text, nothing more:"""
            },
            "finding_title": {
                "grammarly": "Refine the following title for clarity and readability while preserving its original intent:",
                "llama": """You are a copy-editor. Improve clarity, flow, grammar, and professionalism while preserving meaning and any URLs/HTML tags.
    OUTPUT RULES:
    - Output ONLY the revised text.
    - Do NOT add any explanations, headings, labels, or lead-ins (e.g., “Here is the edited text:”).
    - Do NOT wrap the output in quotes or Markdown/code fences.
    - Keep all HTML tags that are present in the input unless they are clearly broken.

    Use these Strict instructions:

    - Improve the finding title while keeping changes minimal.
    - Do not add words like "Vulnerability Assessment" or change the title format unnecessarily.
    - Do not introduce new terms or change their meaning.
    - Capitalize the first letter of each word.
    Return the text and nothing more:"""
            },
            "finding_body": {
                "grammarly": "Make this text coherent and fix any grammar issues:",
                "llama": """You are a copy-editor. Improve clarity, flow, grammar, and professionalism while preserving meaning and any URLs/HTML tags.

    OUTPUT RULES:
    - Output ONLY the revised text.
    - Do NOT add any explanations, headings, labels, or lead-ins (e.g., “Here is the edited text:”).
    - Do NOT wrap the output in quotes or Markdown/code fences.
    - Keep all HTML tags that are present in the input unless they are clearly broken.

    Use these Strict instructions:
    - Do not add, remove, or rewrite sentences.
    - Do not invent any new information or make assumptions.
    - Only fix grammar, punctuation, and typos.
    - Remove roundabout phrases like: "during this engagement", "in order to", etc.
    - If a number is written as "five (5)", convert it to just "five". Do not change actual dates.
    - Do not add or remove any HTML tags, including <code>...</code> or <span>...</span>.
    - Keep company names capitalized as: PentestCompany
    - Replace verbose technical phrases with standard abbreviations, **only if they already exist in the text**.

    Return only the edited version of the input text, nothing more:"""
            }
        }

        # --- Process Executive Summary (Sequential) ---
        modified_exec_summary = []
        for field_execsummary_narrative in self.report_content.get("exec_summary", {}).get("custom_fields", []):
            modified_field = field_execsummary_narrative.copy()

            edits = get_processed_edits(
                field_execsummary_narrative.get("text", ""),
                self,
                use_llama=use_llama,
                use_grammarly=use_grammarly,
                prompts=prompts["exec_summary"],
                use_html_wrapping=self.use_html_wrapping
            )

            modified_field["text"] = "".join(edits)
            modified_exec_summary.append(modified_field)

        # --- Helper to process a single finding ---
        def process_single_finding(finding):
            modified_finding = finding.copy()

            title_edits = self.copy_editor.get_edits(
                finding.get("title", ""),
                use_llama=use_llama,
                use_grammarly=use_grammarly,
                prompts=prompts["finding_title"]
            )

            description_edits = get_processed_edits(
                finding.get("description", ""),
                self,
                use_llama=use_llama,
                use_grammarly=use_grammarly,
                prompts=prompts["finding_body"],
                use_html_wrapping=self.use_html_wrapping
            )

            recommendations_edits = get_processed_edits(
                finding.get("recommendations", ""),
                self,
                use_llama=use_llama,
                use_grammarly=use_grammarly,
                prompts=prompts["finding_body"],
                use_html_wrapping=self.use_html_wrapping
            )

            guidance_edits = get_processed_edits(
                finding.get("fields", {}).get("guidance", {}).get("value", ""),
                self,
                use_llama=use_llama,
                use_grammarly=use_grammarly,
                prompts=prompts["finding_body"],
                use_html_wrapping=self.use_html_wrapping
            )

            reproduction_edits = get_processed_edits(
                finding.get("fields", {}).get("reproduction_steps", {}).get("value", ""),
                self,
                use_llama=use_llama,
                use_grammarly=use_grammarly,
                prompts=prompts["finding_body"],
                use_html_wrapping=self.use_html_wrapping
            )

            modified_finding.update({
                "title": "".join(title_edits),
                "description": "".join(description_edits),
                "recommendations": "".join(recommendations_edits)
            })
            if "guidance" in modified_finding.get("fields", {}):
                modified_finding["fields"]["guidance"]["value"] = "".join(guidance_edits)
            if "reproduction_steps" in modified_finding.get("fields", {}):
                modified_finding["fields"]["reproduction_steps"]["value"] = "".join(reproduction_edits)

            return modified_finding

        # --- Process Findings (Parallel!) ---
        modified_findings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_single_finding, finding) for finding in self.report_findings_content]
            for future in concurrent.futures.as_completed(futures):
                modified_findings.append(future.result())

        # Sort back to original order if needed
        modified_findings = sorted(modified_findings, key=lambda x: x["flaw_id"])

        # --- Save the suggestions ---
        self.suggestedfixes_from_llm = {
            "executive_summary_custom_fields": {"custom_fields": modified_exec_summary},
            "findings": modified_findings
        }

        self.retrieved_suggestedfixes_from_llm = True
        print("LLM and Grammarly suggestions retrieved successfully.")
        time.sleep(2)

    def generate_executive_summary(
        self,
        template_path: str,
        use_llama: bool = False,
        use_grammarly: bool = False,
        selected_types: list[str] | None = None,
        extra_ctx: dict | None = None,
    ):
        """
        Generate multi-section, executive-level summary using selected pentest types
        and a parameterized template. Returns list[{'id','label','text'}].

        This version logs each LLM interaction via log_llm_interaction(...).
        """
        # --- load template ---
        with open(template_path, "r", encoding="utf-8") as fh:
            template_sections = yaml.safe_load(fh) or []

        # --- context ---
        findings = self.report_findings_content or []
        selected_types = selected_types or []
        rows = [_finding_row(f) for f in (self.report_findings_content or [])]

        base_ctx = {
            "CLIENT_NAME": self.client_name or "",
            "REPORT_NAME": self.report_name or "",
            "VENDOR_NAME": getattr(self, "vendor_name", "PentestCompany"),
            "FOOTER_VENDOR": getattr(self, "footer_vendor_name", "IntrusionOps"),
            "TEST_TYPES": ", ".join(s.title() for s in selected_types) or "General",
            "TEST_TYPES_ENGLISH": _english_test_types(selected_types),
            "REPORT_MONTH_YEAR": datetime.now().strftime("%B %Y"),
            "FINDINGS_ROWS": "\n".join(rows),
        }
        if extra_ctx:
            base_ctx.update({k: v for k, v in extra_ctx.items() if v is not None})

        # Only keep the sections we need (always header+footer; middle based on selection)
        wanted_ids = {"header", "footer"} | {t for t in selected_types if t in {"external","internal","webapp","mobile"}}
        template_sections = [sec for sec in template_sections if (sec.get("id") in wanted_ids)]

        guidance_map = {
            "external": "Emphasize internet-facing exposure and perimeter controls.",
            "internal": "Emphasize lateral movement constraints and segmentation.",
            "webapp":   "Emphasize auth/session, input handling, data exposure, tenant isolation.",
            "mobile":   "Emphasize local storage, transport security, secret handling.",
            "cloud":    "Emphasize IAM least-privilege, public exposure, key management, misconfig drift.",
            "social":   "Emphasize phishing exposure, user behavior, MFA resilience.",
        }
        type_guidance = " ".join(guidance_map[t] for t in selected_types if t in guidance_map)

        base_ctx = {
            "CLIENT_NAME": self.client_name or "",
            "REPORT_NAME": self.report_name or "",
            "TEST_TYPES": ", ".join(s.title() for s in selected_types) or "General",
            "TYPE_GUIDANCE": type_guidance,
            "FINDINGS_ROWS": "\n".join(rows),
        }
        if extra_ctx:
            base_ctx.update({k: v for k, v in extra_ctx.items() if v is not None})

        # model string for logging
        model_str = ", ".join(
            m for (m, enabled) in [
                ("llama3:8b-instruct-fp16", use_llama),
                ("grammarly/coedit-xl", use_grammarly),
            ] if enabled
        ) or "none"

        # optional: per-backend kwargs
        llama_kwargs = {"model_name": "llama3:8b-instruct-fp16", "temperature": 0.2}
        grammarly_kwargs = {"model_name": "grammarly/coedit-xl", "temperature": 0.3, "batch_size": 4}

        # prompts passed into your CopyEditor.get_edits()
        edit_prompts = {
            "llama": (
                "Write the requested section based on the instructions and context below. "
                "Return ONLY the final section text—no preface, no bullets unless asked, no code fences."
            ),
            "grammarly": (
                "Lightly copy-edit the following text for clarity and grammar. "
                "Return ONLY the final text, with no preamble or commentary."
            ),
        }

        # --- generate & log ---
        out_sections = []
        for sec in template_sections:
            sid   = sec.get("id") or "section"
            label = sec.get("label") or sid
            sys_p = sec.get("system_prompt") or ""

            # Final composed prompt shown to the model (we pass as 'text' to get_edits)
            full_prompt = _replace_placeholders(sys_p, base_ctx)

            # Call your CopyEditor path
            try:
                edited_list = self.copy_editor.get_edits(
                    text=full_prompt,
                    use_grammarly=use_grammarly,
                    use_llama=use_llama,
                    grammarly_kwargs=grammarly_kwargs,
                    llama_kwargs=llama_kwargs,
                    prompts=edit_prompts,
                )
            except Exception as e:
                # still log the attempt
                try:
                    log_llm_interaction(
                        section=sid,
                        field_id=None,
                        model=model_str,
                        prompt=full_prompt,
                        response=f"[ERROR] {e}",
                    )
                except Exception:
                    pass
                edited_list = [""]

            # Normalize CopyEditor output to a string
            if isinstance(edited_list, list) and edited_list:
                first = edited_list[0]
                raw_text = first if isinstance(first, str) else (first.get("text") or first.get("generated_text") or "")
            elif isinstance(edited_list, str):
                raw_text = edited_list
            else:
                raw_text = ""

            # Log raw model output
            try:
                log_llm_interaction(
                    section=sid,
                    field_id=None,
                    model=model_str,
                    prompt=full_prompt,
                    response=raw_text,
                )
            except Exception:
                pass

            # Post-process before saving
            clean = _execsum_scrub_ids(_execsum_postprocess(raw_text))

            out_sections.append({"id": sid, "label": label, "text": clean})

        # --- build ONE combined Executive Summary: header + selected sections + footer ---
        sec_map = {s["id"]: (s.get("label") or s["id"], s.get("text","").strip()) for s in out_sections}

        def _mk_heading(name: str) -> str:
            line = "-" * len(name)
            return f"{name}\n{line}"

        parts = []
        # header
        if "header" in sec_map and sec_map["header"][1]:
            parts.append(sec_map["header"][1])
        # selected mids (keep order from selected_types)
        for sid in [t for t in selected_types if t in ("external","internal","webapp","mobile")]:
            label, text = sec_map.get(sid, (sid.title(), ""))
            if text:
                parts.append(_mk_heading(label))
                parts.append(text)
        # footer
        if "footer" in sec_map and sec_map["footer"][1]:
            parts.append(sec_map["footer"][1])

        combined_exec = "\n\n".join(parts).strip()

        # --- find the REAL Executive Summary field id from the loaded report ---
        def _get_exec_overview_field_id():
            escf_orig = (
                ((self.report_content or {}).get("exec_summary") or {}).get("custom_fields")
                or ((self.report_content or {}).get("executive_summary_custom_fields") or {}).get("custom_fields")
                or []
            )
            for f in escf_orig:
                label = (f.get("label") or "").lower()
                if "executive" in label and ("summary" in label or "overview" in label):
                    return f.get("id"), f.get("label") or "Executive Summary"
            # fallback: if there is exactly one custom field, use it
            if len(escf_orig) == 1:
                f = escf_orig[0]
                return f.get("id"), f.get("label") or "Executive Summary"
            return None, "Executive Summary"

        target_id, target_label = _get_exec_overview_field_id()

        # --- persist as a SINGLE suggested item bound to that id (nil-safe) ---
        if not isinstance(getattr(self, "suggestedfixes_from_llm", None), dict):
            self.suggestedfixes_from_llm = {}
        escf = self.suggestedfixes_from_llm.get("executive_summary_custom_fields")
        if not isinstance(escf, dict):
            escf = {"custom_fields": []}
            self.suggestedfixes_from_llm["executive_summary_custom_fields"] = escf

        # escf points to self.suggestedfixes_from_llm.setdefault("executive_summary_custom_fields", {})
        existing = escf.get("custom_fields")
        if not isinstance(existing, list):
            existing = [existing] if isinstance(existing, dict) else []
            escf["custom_fields"] = existing

        def _norm(s: str) -> str:
            return " ".join((s or "").lower().split())

        payload = {
            "id":   target_id or "executive_summary_overview",
            "label": target_label,  # e.g., "Executive Summary Overview"
            "text":  combined_exec,
        }

        # find by id first; fallback to normalized label match
        idx = next(
            (i for i, cf in enumerate(existing)
             if (target_id and cf.get("id") == target_id) or _norm(cf.get("label")) == _norm(target_label)),
            None
        )

        if idx is not None:
            existing[idx].update(payload)       # update just the exec overview entry
        else:
            existing.append(payload)            # add alongside Scope or others


        self.suggestedfixes_from_llm.setdefault("findings", [])
        self.retrieved_suggestedfixes_from_llm = True

        # return what we just saved for downstream UI
        return escf["custom_fields"]


    # --------------------------------------------------------------------------------------------


    # Report update functions for suggestions retrieved from LLM
    # ---------------------------------------------------------------------------------------------

    @authentication_required_decorator
    def update_executive_summary(self, field_id, updated_text):
        """
        Update a specific executive summary field in PlexTrac and, on success,
        log *all* exec summary fields (original+modified) so the audit log is complete.
        """
        client_id = self.client_id
        report_id = self.report_id

        if not client_id or not report_id:
            print("Error: Client ID or Report ID is missing.")
            return False

        if not self.report_content:
            print("Error: Full report content is missing. Fetch it before updating.")
            return False

        existing_custom_fields = self.report_content.get("exec_summary", {}).get("custom_fields", [])
        if not isinstance(existing_custom_fields, list):
            print("Error: exec_summary.custom_fields is missing or not a list.")
            return False

        # ----- 1) Precompute originals + would-be updates (no mutation yet) -----
        audit_entries = []            # list of dicts to feed log_change() after success
        modified_custom_fields = []   # list for self.report_content mutation
        field_updated = False
        target_id_str = str(field_id)

        for f in existing_custom_fields:
            fid = str(f.get("id"))
            original_text = f.get("text", "")

            if fid == target_id_str:
                # This is the one we are updating
                new_text = updated_text
                field_updated = True

                mf = f.copy()
                mf["text"] = updated_text
                modified_custom_fields.append(mf)
                audit_entries.append({
                    "section": "executive_summary",
                    "field_id": fid,
                    "original": original_text,
                    "modified": new_text,
                    "accepted": True,
                })
            else:
                # Not touched
                new_text = original_text
                modified_custom_fields.append(f)

        if not field_updated:
            print(f"Error: No executive summary field found with ID {field_id}.")
            return False

        # ----- 2) Mutate local copy that we'll send to the server -----
        self.report_content["exec_summary"]["custom_fields"] = modified_custom_fields
        updated_report = self.report_content.copy()

        # ----- 3) PUT update first -----
        url = f"{self.base_url}/client/{client_id}/report/{report_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

        try:
            response = self.make_request_authenticated("PUT", url, headers=headers, json=updated_report)

            # ----- 4) Only log after a successful update -----
            if response.status_code == 200:
                for e in audit_entries:
                    log_change(
                        e["section"],
                        e["field_id"],
                        e["original"],
                        e["modified"],
                        e["accepted"]
                    )
                print(f"Executive summary field {field_id} updated successfully.")
                time.sleep(3)
                return True
            else:
                print(f"Error updating field {field_id}: {response.status_code}, {response.text}")
                time.sleep(3)
                return False

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(3)
            return False


    @authentication_required_decorator
    def update_finding(
        self,
        finding_id,
        updated_title=None,
        updated_description=None,
        updated_recommendations=None,
        updated_guidance=None,
        updated_reproduction_steps=None,
    ):
        """
        Update a specific finding in PlexTrac and, on success, log only the fields
        the user actually accepted (i.e., the ones passed in and changed).
        """
        client_id = self.client_id
        report_id = self.report_id

        if not client_id or not report_id:
            print("Error: Client ID or Report ID is missing.")
            time.sleep(3)
            return False

        findings = self.report_findings_content if original else ((self.suggestedfixes_from_llm or {}).get("findings") or [])

        # ---- find target finding ----
        target = None
        for f in findings:
            if str(f.get("id")) == str(finding_id):
                target = f
                break

        if target is None:
            print(f"Error: No finding found with ID {finding_id}.")
            time.sleep(3)
            return False

        # ---- snapshot originals (strings only) BEFORE mutation ----
        def _nested(d, *path, default=""):
            cur = d
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    return default
                cur = cur[p]
            return cur

        originals = {
            "title": target.get("title", ""),
            "description": target.get("description", ""),
            "recommendations": target.get("recommendations", ""),
            "guidance": _nested(target, "fields", "guidance", "value", default=""),
            "reproduction_steps": _nested(target, "fields", "reproduction_steps", "value", default=""),
        }

        # ---- build list of accepted updates (only if param is provided AND changed) ----
        updates = []  # each: (log_label, path_key, new_value, original_value)
        if updated_title is not None and updated_title != originals["title"]:
            updates.append(("finding title", ("title",), updated_title, originals["title"]))
        if updated_description is not None and updated_description != originals["description"]:
            updates.append(("finding description", ("description",), updated_description, originals["description"]))
        if updated_recommendations is not None and updated_recommendations != originals["recommendations"]:
            updates.append(("finding recommendations", ("recommendations",), updated_recommendations, originals["recommendations"]))
        if updated_guidance is not None and updated_guidance != originals["guidance"]:
            updates.append(("finding guidance", ("fields", "guidance", "value"), updated_guidance, originals["guidance"]))
        if updated_reproduction_steps is not None and updated_reproduction_steps != originals["reproduction_steps"]:
            updates.append(("finding reproduction_steps", ("fields", "reproduction_steps", "value"), updated_reproduction_steps, originals["reproduction_steps"]))

        if not updates:
            print("No changes to apply for this finding.")
            time.sleep(2)
            return False

        # ---- make a deep copy to avoid mutating nested dicts in the original ----
        modified = deepcopy(target)

        # ---- apply accepted updates to the deep copy ----
        for _, path, new_val, _orig in updates:
            if len(path) == 1:
                modified[path[0]] = new_val
            else:
                cur = modified
                for k in path[:-1]:
                    if k not in cur or not isinstance(cur[k], dict):
                        cur[k] = {}
                    cur = cur[k]
                cur[path[-1]] = new_val

        # ---- PUT the single finding update ----
        url = f"{self.base_url}/client/{client_id}/report/{report_id}/flaw/{finding_id}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

        try:
            response = self.make_request_authenticated("PUT", url, headers=headers, json=modified)
            if response.status_code == 200:
                # update in-memory finding (so UI reflects new text)
                for i, f in enumerate(self.report_findings_content):
                    if str(f.get("id")) == str(finding_id):
                        self.report_findings_content[i] = modified
                        break

                # ---- log ONLY the accepted updates after success ----
                for log_label, _path, new_val, orig_val in updates:
                    log_change(log_label, finding_id, orig_val, new_val, accepted=True)

                print(f"Finding {finding_id} updated successfully.")
                time.sleep(2)
                return True
            else:
                print(f"Error updating finding {finding_id}: {response.status_code}, {response.text}")
                time.sleep(3)
                return False

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(3)
            return False


    # ---------------------------------------------------------------------------------------------

    # Report Display Functions
    # These functions can be called after all of the necessary report items are downloaded
    # --------------------------------------------------------------------------------------------------------------
    def get_local_exec_summary(self, original=True):
        """Extract the executive summary from a report and return it. - suitable for display in curses"""
        executive_summary = []
        if self.report_content is not None:
            if self.report_content.get("exec_summary"):
                # Check if 'custom_fields' exists and is a list
                custom_fields = self.report_content.get("exec_summary").get('custom_fields', []) if original is True else self.suggestedfixes_from_llm["executive_summary_custom_fields"]["custom_fields"]
                if custom_fields:
                    for field in custom_fields:
                        executive_summary.append( field.get('label', 'No label available')+'\n----------------------\n'+
                                                  field.get('text', 'No text available')+"\n")
                else:
                    print(f"No custom fields found in the executive summary.")
            else:
                print(f"No executive summary found.")
        return executive_summary

    
    def get_local_findings(self, original=True):
        """ return local copy of findings, original or llm generated - suitable for display in curses """
        report_findings_content=[]

        findings=self.report_findings_content if original is True else self.suggestedfixes_from_llm["findings"]
        if findings is not None:
            for f in findings:
                report_findings_content.append( f"Title: {f['title']}\n" +
                                                '---------------------------------------------\n\n' +
                                                f"\n*Description: {f['description']}\n\n" +
                                                f"\n\n*Recommendations: {f['recommendations']}\n\n" +
                                                f"\n\n*Guidance: {f['fields'].get('guidance', {}).get('value', '')}\n\n" +
                                                f"\n\n*Reproduction Steps: {f['fields'].get('reproduction_steps', {}).get('value', '')}\n\n"
                                             )
        return report_findings_content

    def generate_visual_reportdiff(self):
        """Generate sentence-based diffs for exec summary + findings.
        Safe when only some suggestions exist (mirrors originals if modified is missing).
        Also builds self.visual_diff_chunks for per-section/per-finding views.
        """
        exec_summary_diffs = []
        report_findings_diffs = []
        self.visual_diff_chunks = {}

        def sentence_diff(original, modified):
            """Sentence-based diff. Keeps periods and whitespace splitting similar to display."""
            diff = difflib.ndiff(
                re.split(r'(?<=\.)\s*', (original or "").strip()),
                re.split(r'(?<=\.)\s*', (modified or "").strip())
            )
            out = []
            for token in diff:
                if token.startswith("+ "):
                    out.append(("add", token[2:]))
                elif token.startswith("- "):
                    out.append(("remove", token[2:]))
                else:
                    out.append(("normal", token[2:]))
            return out


        # ---------- Executive Summary ----------
        # Originals from the loaded report
        orig_exec = (
            ((self.report_content or {}).get("exec_summary") or {}).get("custom_fields")
            or ((self.report_content or {}).get("executive_summary_custom_fields") or {}).get("custom_fields")
            or []
        )

        # Proposed (merged overlay): originals with suggestions applied by field id
        # merge_exec_fields_for_views(report_content, suggestions) -> List[dict]
        mod_exec = merge_exec_fields_for_views(self.report_content, self.suggestedfixes_from_llm)

        # If nothing suggested yet, mirror originals (so diff shows "no changes")
        if not mod_exec:
            mod_exec = [dict(cf) for cf in orig_exec]

        def _cf_key(cf, fallback_idx=None):
            return str(
                cf.get("id")
                or cf.get("field_id")
                or cf.get("label")
                or (f"idx:{fallback_idx}" if fallback_idx is not None else "unk")
            )

        # Preserve original order; include any extras from modified (usually none)
        orig_keys = [_cf_key(cf, i) for i, cf in enumerate(orig_exec)]
        mod_keys  = [_cf_key(cf, i) for i, cf in enumerate(mod_exec)]
        ordered_keys = []
        seen = set()
        for k in orig_keys + mod_keys:
            if k not in seen:
                ordered_keys.append(k); seen.add(k)

        # Lookup maps
        orig_map = {_cf_key(cf, i): cf for i, cf in enumerate(orig_exec)}
        mod_map  = {_cf_key(cf, i): cf for i, cf in enumerate(mod_exec)}

        for idx, key in enumerate(ordered_keys):
            cf_o = orig_map.get(key, {}) or {}
            cf_m = mod_map.get(key, {}) or {}

            # Prefer the original label when available
            label = (cf_o.get("label") or cf_m.get("label") or "Unknown Section").strip()
            text_o = (cf_o.get("text") or "").strip()
            text_m = (cf_m.get("text") or "").strip()

            diffs = sentence_diff(text_o, text_m)
            # If you want to skip unchanged sections in the visual list, guard here:
            # if not diffs: continue

            exec_summary_diffs.append(("title", f"=== {label} ==="))
            exec_summary_diffs.extend(diffs)
            exec_summary_diffs.append(("normal", ""))

            # chunk for focused/full views
            self.visual_diff_chunks[f"exec_summary_{idx}"] = [("title", f"=== {label} ===")] + diffs


        # ---------- Findings ----------
        orig_findings = self.report_findings_content or []
        mod_findings = (self.suggestedfixes_from_llm or {}).get("findings")
        if mod_findings is None:
            # Mirror originals when no suggestions exist for findings
            mod_findings = [dict(f) for f in orig_findings]

        def _fid(f, fallback_idx=None):
            return str(f.get("id") or f.get("finding_id") or f.get("flaw_id") or f.get("title") or (f"idx:{fallback_idx}" if fallback_idx is not None else "unk"))

        orig_fkeys = []
        for i, f in enumerate(orig_findings):
            orig_fkeys.append(_fid(f, i))
        mod_fkeys = []
        for i, f in enumerate(mod_findings):
            mod_fkeys.append(_fid(f, i))

        seen_f = set()
        ordered_fkeys = []
        for k in orig_fkeys + mod_fkeys:
            if k not in seen_f:
                ordered_fkeys.append(k)
                seen_f.add(k)

        orig_fmap = {_fid(f, i): f for i, f in enumerate(orig_findings)}
        mod_fmap  = {_fid(f, i): f for i, f in enumerate(mod_findings)}

        def _get_field(obj, name):
            if name in ("guidance", "reproduction_steps"):
                return (obj or {}).get("fields", {}).get(name, {}).get("value", "") or ""
            return (obj or {}).get(name, "") or ""

        sections = ["title", "description", "recommendations", "guidance", "reproduction_steps"]

        for find_idx, fkey in enumerate(ordered_fkeys):
            f_o = orig_fmap.get(fkey, {}) or {}
            f_m = mod_fmap.get(fkey, {}) or {}

            title_o = f_o.get("title", "") or "Untitled Finding"
            report_findings_diffs.append(("title", f"=== {title_o} ==="))

            for section in sections:
                report_findings_diffs.append(("section", f"{section.capitalize()}:"))
                text_o = _get_field(f_o, section)
                text_m = _get_field(f_m, section)

                diffs = sentence_diff(text_o, text_m)
                report_findings_diffs.extend(diffs)
                report_findings_diffs.append(("normal", ""))

                self.visual_diff_chunks[f"finding_{find_idx}_{section}"] = [("section", f"{section.capitalize()}:")] + diffs

        # ---------- Store combined ----------
        self.visual_diff = exec_summary_diffs + report_findings_diffs
        self.visual_diff_generated = True
        return self.visual_diff
# --------------------------------------------------------------------------------------------------------------

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



### 
### Logging Functions
##############################################################################################################

def log_change(section, field_id, original_text, modified_text, accepted, LOG_FILE="peer_review_log.json"):
    """Log accepted/rejected changes during peer review."""
    log_entry = {
        "section": section,
        "field_id": field_id,
        "original": original_text,
        "modified": modified_text,
        "accepted": accepted
    }

    try:
        # Load existing log data
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []  # Start fresh if no log exists or file is corrupted

    logs.append(log_entry)

    # Save updated logs
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"Logged change: {log_entry}")

def log_llm_interaction(
    section: str,
    field_id: str | int | None,
    model: str,
    prompt: str,
    response: str,
    LOG_FILE: str = "llm_interaction_log.json",
    max_chars: int = 2000,   # truncate huge prompts/responses
) -> None:
    """
    Append a single LLM call (prompt + response) to a JSON log file.
    Mirrors the structure of `log_change()` for consistency.

    Args:
        section   – 'exec_summary', 'finding_title', etc.
        field_id  – ID of the field/finding when applicable (else None)
        model     – 'grammarly/coedit-large', 'llama3.1:8b‑instruct', etc.
        prompt    – Full prompt sent to the model
        response  – Text returned by the model
    """
    entry = {
        "section": section,
        "field_id": field_id,
        "model": model,
        "prompt": prompt[:max_chars],
        "response": response[:max_chars],
        "timestamp": time.strftime("%Y‑%m‑%d %H:%M:%S"),
    }

    try:
        with open(LOG_FILE, "r") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(entry)

    with open(LOG_FILE, "w") as fh:
        json.dump(data, fh, indent=4)

    print(f"Logged LLM interaction for section '{section}', field {field_id}.")
# ------------------------------------------------------------------------------------------


def load_template_execsummary(path: str) -> list[dict]:
    """
    Reads a YAML or JSON executive‑summary template file and returns
    it as a list of section‑definition dictionaries.

    Raises FileNotFoundError if the file is missing and ValueError
    for unsupported extensions.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in (".yml", ".yaml"):
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or []
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    else:
        raise ValueError("Template must be .yml/.yaml or .json")


def usage():
    """Prints usage instructions for the PlexTrac CLI Tool."""
    print("""
    Usage: python your_script.py --server-fqdn <FQDN> --client-name <CLIENT> --report-name <REPORT> [options]

    Required arguments:
      --server-fqdn     The Fully Qualified Domain Name (FQDN) of the PlexTrac server.
      --client-name     The name of the client in PlexTrac.
      --report-name     The name of the report to process.

    Optional arguments:
      --use-llama       Enable Llama for text processing.
      --use-grammarly   Enable Grammarly for text processing.
      --tgi-server-url  TGI Server URL for Grammarly (default: None) - TGI Server usually listens on 8080
      --ollama-server-url   Ollama server URL (default: None)  - Ollama usually listens on 11434

    Example usage:
      python peer_review_helper.py --server-fqdn example.com --client-name Acme --report-name SecurityAudit --use-llama --ollama-remote-url http://127.0.0.1:11434
    """)
    

def initialize():
    parser = argparse.ArgumentParser(description="PlexTrac CLI Tool", usage=usage.__doc__)
    parser.add_argument("--server-fqdn", required=True, help="PlexTrac server FQDN")
    parser.add_argument("--client-name", required=True, help="Client name")
    parser.add_argument("--report-name", required=True, help="Report name")
    parser.add_argument("--use-llama", action="store_true", help="Enable Llama for text processing (disabled by default)") 
    parser.add_argument("--use-grammarly", action="store_true", help="Enable Grammarly for text processing (disabled by default)")
    parser.add_argument("--tgi-server-url", default=None, help="TGI Server URL for Grammarly (default: None) - TGI Server usually listens on 8080")
    parser.add_argument("--ollama-remote-url", default=None, help="Ollama server URL (default: None) - Ollama usually listens on 11434")
    parser.add_argument("--use-html-aware-editing", action="store_true", help="Preserve and reinsert inline HTML tags when editing (default: disabled)")

    args = parser.parse_args()

    # Prompt for username and password
    username = input("Username: ")
    password = getpass.getpass("Password: ")

    # Initialize API client
    api = PlexTracAPI(args.server_fqdn, username, password, ollama_remote_url=args.ollama_remote_url, tgi_remote_url=args.tgi_server_url, 
                      use_llama=args.use_llama, use_grammarly=args.use_grammarly, use_html_wrapping=args.use_html_aware_editing)
    api.authenticate()
    return args, api

def test_prompt_on_finding(api, finding_index, section_index, prompt, use_llama=True, use_grammarly=False):
    """
    Quickly test a custom prompt on a single section of a finding.
    Shows both raw LLM output and output using tag-safe editing via extractonlytext_sendtollm().
    """
    if not api.report_findings_content:
        print("No findings loaded. Run api.get_report_findings_content() first.")
        return

    if finding_index >= len(api.report_findings_content):
        print("Invalid finding index.")
        return

    finding = api.report_findings_content[finding_index]
    section_map = {
        0: ("title", finding.get("title", "")),
        1: ("description", finding.get("description", "")),
        2: ("recommendations", finding.get("recommendations", "")),
        3: ("guidance", finding.get("fields", {}).get("guidance", {}).get("value", "")),
        4: ("reproduction_steps", finding.get("fields", {}).get("reproduction_steps", {}).get("value", "")),
    }

    section_key, original_text = section_map.get(section_index, (None, None))
    if section_key is None:
        print("Invalid section index.")
        return

    print(f"\n=== Testing {section_key.upper()} on Finding #{finding_index} ===\n")

    # 1. ORIGINAL
    print("----- ORIGINAL TEXT -----")
    print(soft_format_html_for_terminal(original_text))

    # 2. BASIC LLM EDIT
    fixed = api.copy_editor.get_edits(
        original_text,
        use_llama=use_llama,
        use_grammarly=use_grammarly,
        prompts={"llama": prompt, "grammarly": prompt}
    )
    joined_fixed = "".join(fixed)

    print("\n----- LLM OUTPUT (plain text edit only) -----")
    print(soft_format_html_for_terminal(joined_fixed))

    # 3. HTML-PRESERVING EDIT (EXTRACT + REINSERT)
    def _wrapped_edit_func(text_chunk: str) -> str:
        result = api.copy_editor.get_edits(
            text_chunk,
            use_llama=use_llama,
            use_grammarly=use_grammarly,
            prompts={"llama": prompt, "grammarly": prompt}
        )
        return "".join(result)

    fully_wrapped = extractonlytext_sendtollm(original_text, _wrapped_edit_func)

    print("\n----- LLM OUTPUT (with inline HTML handling) -----")
    print(soft_format_html_for_terminal(fully_wrapped))


if __name__ == "__main__":
    args,api=initialize()

    # Backup report first
    if os.path.exists(args.client_name+"_"+args.report_name+'.ptrac') or os.path.exists(args.client_name+"_"+args.report_name+'.docx'):
        print (f"Report already exists and refusing to overwrite: {args.report_name}")
        sys.exit(0)

    exported=api.export_report(args.client_name, args.report_name)
    if exported is False:
        print ("Error exporting reports...quitting")
        sys.exit(0)
    
    # Get original report and display in an curses interface
    api.get_report_findings_content(args.client_name, args.report_name)
    exec_summary=api.get_local_exec_summary()
    findings=api.get_local_findings()
    all_text = exec_summary + findings
    paginated_text=generate_paginated_text(all_text)
    curses.wrapper(interactive_text_viewer, api, paginated_text)
