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

from models import SpacyModel, LlamaModel, CopyEditor, GrammarlyModel, extractonlytext_sendtollm, get_processed_edits
from textwrap import shorten
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher, ndiff

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

    def generate_executive_summary(self, template_path: str, use_llama=True, use_grammarly=False):
        """
        Build a brand‑new executive summary from report_findings_content and
        a YAML/JSON template.  Returns the list of {'label','text', 'id'} dicts.
        """

        def first_n_sentences(text: str, n: int = 3) -> str:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            return " ".join(sentences[:n])

        def format_findings_compact(findings: list[dict], max_chars=12000) -> str:
            blocks = [
                f"{f['title']}"
                for f in findings
            ]
            return "\n".join(blocks)[:max_chars]

        def format_findings_as_report(findings: list[dict], max_chars=12000) -> str:
            blocks = []
            for f in findings:
                title = f.get("title", "Untitled")
                severity = f.get("severity", "Unknown")
                desc = f.get("description", "").strip()
                recs = f.get("recommendations", "").strip()

                block = f"""Title: {title}
        Severity: {severity}
        Description:
        {desc if desc else '(No description provided)'}

        Recommendations:
        {recs if recs else '(No recommendations provided)'}
        ---
        """
                blocks.append(block.strip())

            result = "\n\n".join(blocks)
            return result[:max_chars]

        if not self.report_findings_content:
            raise ValueError("Findings not loaded. Run get_report_findings_content()")

        template = load_template_execsummary(template_path) # small helper to read YAML
        findings_for_prompt = format_findings_compact(self.report_findings_content)

        new_sections = []
        for section in template:
            prompt_template = section["system_prompt"]
            prompt = prompt_template.replace("{{FINDINGS_JSON_SNIPPET_HERE}}", findings_for_prompt)

            generated_text = self.copy_editor.get_edits(
                prompt,                        # treat full prompt as “text” to LLM
                use_llama=use_llama,
                use_grammarly=use_grammarly,
                prompts={"llama": prompt, "grammarly": prompt}
            )[0]                              # join omitted because get_edits returns list

            log_llm_interaction(
                section=section['id'],
                field_id=None,
                model=", ".join(m for m, enabled in [("llama3:8b-instruct-fp16", use_llama), 
                                                     ("grammarly/coedit-xl", use_grammarly)] if enabled),
                prompt=prompt,
                response=generated_text
            )

            new_sections.append({
                "id": section["id"],          # keep deterministic IDs for updates later
                "label": section["label"],
                "text": generated_text.strip()
            })

        return new_sections

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
        orig_exec = (self.report_content or {}).get("exec_summary", {}).get("custom_fields", []) or []
        mod_exec_cont = (self.suggestedfixes_from_llm or {}).get("executive_summary_custom_fields", {}) or {}
        mod_exec = mod_exec_cont.get("custom_fields")
        if mod_exec is None:
            # No modified exec summary yet -> mirror originals to avoid crashing and show "no changes"
            mod_exec = [dict(cf) for cf in orig_exec]

        def _cf_key(cf, fallback_idx=None):
            return str(cf.get("id") or cf.get("field_id") or cf.get("label") or (f"idx:{fallback_idx}" if fallback_idx is not None else "unk"))

        # Order keys: keep original order, then any extra keys from modified
        orig_keys = []
        for i, cf in enumerate(orig_exec):
            orig_keys.append(_cf_key(cf, i))
        mod_keys = []
        for i, cf in enumerate(mod_exec):
            mod_keys.append(_cf_key(cf, i))

        seen = set()
        ordered_keys = []
        for k in orig_keys + mod_keys:
            if k not in seen:
                ordered_keys.append(k)
                seen.add(k)

        # Build maps for quick lookup
        orig_map = {_cf_key(cf, i): cf for i, cf in enumerate(orig_exec)}
        mod_map = {_cf_key(cf, i): cf for i, cf in enumerate(mod_exec)}

        for idx, key in enumerate(ordered_keys):
            cf_o = orig_map.get(key, {}) or {}
            cf_m = mod_map.get(key, {}) or {}

            label = cf_o.get("label") or cf_m.get("label") or "Unknown Section"
            text_o = cf_o.get("text", "") or ""
            text_m = cf_m.get("text", "") or ""

            exec_summary_diffs.append(("title", f"=== {label} ==="))
            diffs = sentence_diff(text_o, text_m)
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


# User Interface 
# ------------------------------------------------------------------------------------------
def interactive_text_viewer(stdscr, pages):
    """Curses-based interactive viewer with text wrapping and vertical scrolling."""
    curses.curs_set(0)
    page_index = 0
    current_view = "ORIGINAL VIEW-"
    scroll_offset = 0  # Track vertical scrolling

    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()

        # Message at the top of the window
        top_message = f"Page {page_index + 1}/{len(pages)} (q-quit, c-gen/regen exec summary, r-get llm suggestions, p-view suggestions, o-view original, d-view diffs, u-import updates to plextrac)"
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
        # Use LLM to generate executive summary from the findings list
        elif key == ord('c'):  # allow on first run and subsequent runs
            try:
                # 1) UI hint
                try:
                    stdscr.addstr(1, 2, "Generating executive summary from LLM…", curses.A_BOLD)
                    stdscr.clrtoeol()
                    stdscr.refresh()
                except Exception:
                    pass

                # 2) Generate sections (uses your existing method + flags)
                sections = api.generate_executive_summary(
                    template_path="templates/execsummary.yml",
                    use_llama=api.use_llama,
                    use_grammarly=api.use_grammarly,
                )

                # 3) Stash into suggestedfixes structure (create container if missing)
                if api.suggestedfixes_from_llm.get("executive_summary_custom_fields") is None:
                    api.suggestedfixes_from_llm["executive_summary_custom_fields"] = {"custom_fields": []}
                api.suggestedfixes_from_llm["executive_summary_custom_fields"]["custom_fields"] = sections

                # 4) Mark suggestions present and rebuild visual diffs
                api.retrieved_suggestedfixes_from_llm = True
                api.generate_visual_reportdiff()

                # 5) Optional: switch to LLM view so user sees it immediately
                current_view = "LLM VIEW-"
                page_index = 0
                scroll_offset = 0

                # Optional: brief success message
                try:
                    stdscr.addstr(1, 2, "Executive summary generated. Press d to see diffs or u to update.", curses.A_BOLD)
                    stdscr.clrtoeol()
                    stdscr.refresh()
                except Exception:
                    pass

            except Exception as e:
                # Non-fatal: show error on the banner line
                try:
                    stdscr.addstr(1, 2, f"Exec summary generation failed: {e}", curses.A_BOLD)
                    stdscr.clrtoeol()
                    stdscr.refresh()
                except Exception:
                    pass


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


# ==== HUNK UTILITIES ======================================================

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
    curses.wrapper(interactive_text_viewer, paginated_text)
