# Tool to assist with peer reviewing of pentest reports

NLP-powered CLI/TUI helper for PlexTrac reports. It fetches a report, generates LLM-based suggestions (exec summary + findings), shows readable diffs, and lets you accept/skip/edit changes **per change**, with audit logging.  Thanks to Danny Nassre (nassre@gmail.com) for supplying the LLM integration.

# Todos

* Automatically generate the executive summary given a list of findings (In progress)
* Compare the findings to the executive summary to ensure exec summary has the right wording (In progress)
* Automatically generate findings given a directory of raw findings data

## Features

* **LLM suggestions**

  * Findings: grammar/clarity improvements with your copy-editing pipeline (Ollama + Grammarly-style, serial).
  * **Executive Summary Generator (`c`)**: produce structured sections from findings + template. 
* **Readable diffs**

  * Sentence/word diff, wrapped lines, scrollable view (`d`). 
* **Per-change Update Mode (`u`)**

  * Review each hunk/change inside a section (exec summary first, then each finding field).

    * `a` accept, `s` skip, `e` edit in `$EDITOR` (vi) then accept. 
* **Multiple views**

  * `o` Original, `p` LLM suggestions, `d` Diffs, `u` Update mode. 
* **HTML-aware editing (optional)**

  * Strip/reinsert inline tags to preserve formatting.  This has not been further developed so results may vary.
* **Audit logging**

  * Logs accepted/edited changes when updates succeed.
* **Achitecture & workflow diagrams**

  * See [`architecture.md`](./assets/architecture.md). 

## Requirements

* Python 3.10
* A PlexTrac account with API access
* (Optional) **Ollama** running locally or remote (default `127.0.0.1:11434`)
* (Optional) **Grammarly-style** model via TGI server (default `127.0.0.1:8080`) or local HF pipeline

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration

CLI flags let you choose backends:

* `--use-llama` Enable Llama (Ollama local/remote)
* `--use-grammarly` Enable Grammarly-style TGI or HF pipeline
* `--tgi-server-url` TGI endpoint (e.g. [http://127.0.0.1:8080](http://127.0.0.1:8080))
* `--ollama-remote-url` Ollama endpoint (e.g. [http://127.0.0.1:11434](http://127.0.0.1:11434))
* `--use-html-aware-editing` Preserve and reinsert inline HTML tags during edits

## Usage

1. **Start the CLI** (replace with your server/client/report):

```bash
python peer_review_helper.py \
  --server-fqdn https://your.plextrac.tld \
  --client-name "Acme Corp" \
  --report-name "Q4 External + WebApp Pentest" \
  --use-llama \
  --use-grammarly
```

2. **Load report**
   The tool authenticates and downloads the executive summary + findings.

3. **(Optional) Generate exec summary (`c`)**

* You’ll be prompted for test types (External/Internal/WebApp/Mobile/etc.).
* The tool composes a structured executive summary using your template and LLMs, then stores it in the “suggested” view. 

4. **(Optional) Run suggestions on findings (`r`)**

* Sends findings text to the copy-editing pipeline; results appear in “LLM view”.

5. **Review**

* **Original view (`o`)**: report as currently stored on PlexTrac.
* **LLM view (`p`)**: proposed text from LLMs.
* **Diff view (`d`)**: scrollable, colorized adds/removes, wrapped lines. 

6. **Update mode (`u`) – per change**

* For each section (exec summary first, then each finding field):

  * Navigate each **hunk** (change) and choose:

    * `a`: accept this change
    * `s`: skip (reject) this change
    * `e`: edit in vi, then accept
    * `n/p`: cycle thru changes without accepting/rejecting
  * When done with a section, press **Enter** to apply updates to PlexTrac.
  * Only accepted/edited hunks are PUT and logged. 

### Keybindings (quick reference)

| Key     | Mode       | Action                                                |
| ------- | ---------- | ----------------------------------------------------- |
| `o`     | any        | Original view (current PlexTrac text)                 |
| `p`     | any        | LLM view (suggested text)                             |
| `d`     | any        | Diff view (scrollable)                                |
| `r`     | any        | Run LLM suggestions for findings                      |
| `c`     | any        | Generate Executive Summary sections via LLMs          |
| `shift+c`| any        | Verify exec summary matches findings                 |
| `u`     | any        | Enter Update mode (per-change apply)                  |
| `a`     | update     | Accept current hunk/change                            |
| `s`     | update     | Skip current hunk/change                              |
| `e`     | update     | Edit current hunk in vi then accept                   |
| `n/p`   | update     | View next/previous hunk                               |
| `v`     | update     | View the full set of changes for the current section  |
| `f`     | update     | View just the current selected hunk                   |
| `Enter` | update     | Apply accepted/edited changes for this section (PUT)  |
| `q`     | any/update | Quit current view / leave update early                |

## Templates

Executive summary templates live in `templates/execsummary.yml`. The generator fills placeholders using the current findings + your chosen test types, then runs the LLM pipeline per section before placing text into the suggested exec-summary fields. 

## Diagrams

* **Architecture** and **Review workflow** diagrams are in [`architecture.md`](./assets/architecture.md).

## Troubleshooting

* If you don’t see diffs, ensure you’ve generated suggestions (`r` or `c`) and the visual diff has been rebuilt.
* If `vi` opens for edit, after saving/exiting the editor, curses mode is restored automatically and the edit is applied to the current hunk/section. 
