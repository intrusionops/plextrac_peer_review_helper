The below diagrams show the architecture of the tool and its review workflow:

```mermaid
%%{init: {
  "theme": "dark",
  "themeVariables": {
    "background": "transparent",
    "primaryTextColor": "#e5e7eb",
    "lineColor": "#93c5fd",
    "fontFamily": "Arial, Helvetica, sans-serif"
  },
  "flowchart": {
    "htmlLabels": false,
    "nodeSpacing": 45,
    "rankSpacing": 55,
    "useMaxWidth": true
  }
}}%%
flowchart TD
  U["User in Terminal"] -->|runs CLI| CLI["peer_review_helper.py (CLI &#43; Curses UI)"]

  subgraph PT["PlexTrac"]
    PTAPI["PlexTrac REST API"]
    DB["Reports &amp; Findings"]
    PTAPI --- DB
  end

  subgraph CORE["Peer Review Helper - Core"]
    API["PlexTracAPI<br/>(auth, token refresh, GET/PUT, export)"]
    DIFF["Diff Builder<br/>(sentence/word diffs)"]
    LOG["Audit Log<br/>(peer_review_log.json)"]
    API --> DIFF
    API --> LOG
  end

  subgraph NLP["Copy Editing Pipeline"]
    CE["CopyEditor<br/>(serial pipeline)"]
    SPACY["spaCy<br/>(tokenization)"]
    HTML["HTML-aware editing<br/>(tag strip/reinsert)"]
    CE --> SPACY
    CE --> HTML

    subgraph LLMs["Model Backends"]
      LLAMA["Llama via Ollama<br/>(local or remote)"]
      GRAM["Grammarly-style model<br/>(TGI or HF pipeline)"]
    end
    CE --> LLAMA
    CE --> GRAM
  end

  CLI --> API
  CLI --> CE
  CE -->|suggestions| CLI
  CLI --> DIFF
  DIFF -->|visual chunks| CLI
  CLI -->|accept / skip / edit| API
  API -->|on&nbsp;success| LOG

  PTAPI <--> API

  %% Colors
  classDef cli fill:#1d4ed8,stroke:#93c5fd,color:#f9fafb;
  classDef api fill:#6d28d9,stroke:#c4b5fd,color:#f9fafb;
  classDef diff fill:#065f46,stroke:#34d399,color:#ecfdf5;
  classDef log fill:#b45309,stroke:#fbbf24,color:#fff7ed;
  classDef llm fill:#0e7490,stroke:#67e8f9,color:#ecfeff;
  classDef nlp fill:#166534,stroke:#86efac,color:#ecfdf5;
  classDef pt fill:#991b1b,stroke:#fecaca,color:#fff1f2;

  class CLI cli;
  class API api;
  class DIFF diff;
  class LOG log;
  class LLAMA,GRAM llm;
  class CE,SPACY,HTML nlp;
  class PTAPI,DB pt;
```






This diagram shows the review workflow:


```mermaid
%%{init: {
  "theme": "dark",
  "themeVariables": {
    "background":"transparent",
    "primaryTextColor":"#e5e7eb",
    "fontFamily": "Arial, Helvetica, sans-serif"
  }
}}%%
sequenceDiagram
  autonumber
  participant User as User
  participant CLI as peer_review_helper (CLI/TUI)
  participant API as PlexTracAPI
  participant CE as CopyEditor
  participant PT as PlexTrac REST

  rect rgba(37,99,235,0.15)
    User->>CLI: Launch with server/client/report
    CLI->>API: authenticate()
    API->>PT: POST /auth
    PT-->>API: token
    API-->>CLI: OK (start/refresh token thread)
  end

  rect rgba(34,197,94,0.18)
    CLI->>API: get_full_report_content()
    API->>PT: GET report + findings
    PT-->>API: payload
    API-->>CLI: data loaded
  end

  rect rgba(245,158,11,0.22)
    User->>CLI: Press r (run suggestions)
    CLI->>CE: Exec summary + findings text
    CE->>CE: Llama → Grammarly (serial)
    CE-->>CLI: Suggested edits
  end

  CLI->>CLI: Build sentence hunks + visual diffs
  User->>CLI: Open diff (d) to inspect

  User->>CLI: Enter Update Mode (u)
  loop For each Section (Exec Field or Finding Section)
    Note over CLI: Row 0=menu • Row 1=title • Row 2+=scrollable

    par View toggle
      User->>CLI: Press v → Full diff view
      User->>CLI: Press f → Focus view (hunk + context)
    and Navigation
      User->>CLI: ↑/↓ scroll • PgUp/PgDn page
      User->>CLI: n/p next/prev hunk (focus)
    end

    loop For each Hunk (change)
      alt a = accept
        User->>CLI: Press a
        CLI->>CLI: Mark hunk = accepted
        CLI->>CLI: Recompute staged_text
      else s = skip
        User->>CLI: Press s
        CLI->>CLI: Mark hunk = rejected
        CLI->>CLI: Recompute staged_text
      else e = edit+accept
        User->>CLI: Press e
        CLI->>CLI: Open $EDITOR with hunk text
        CLI->>CLI: Mark hunk = edited
        CLI->>CLI: Update hunk text + recompute staged_text
      end
    end

    alt Enter = apply section
      User->>CLI: Press Enter
      CLI->>API: update_exec_summary()/update_finding() with staged_text
      API->>PT: PUT updated content
      PT-->>API: 200 OK
      API-->>CLI: Success
      CLI->>CLI: log_change() for accepted/edited hunks
    else q = quit section
      User->>CLI: Press q
      CLI-->>API: No PUT
      CLI-->>CLI: No logging
    end
  end

  User-->>CLI: Quit

```
