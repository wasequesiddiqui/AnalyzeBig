# Code Walkthrough - reusable update prompt

Paste this file (or just say **"update the code walkthrough"**) when you want
`code-walkthrough.html` regenerated after adding a folder, adding a `.py` file, or
changing code that the annotations point at.

This document is the operating manual for `_walkthrough/`. It contains no conversation
history - only what you need to reproduce the job.

---

## 1. What exists

| Path | Role |
|---|---|
| `code-walkthrough.html` | **Output.** One self-contained page, six folder tabs, no CDN, works offline. Never edit by hand. |
| `_walkthrough/config.json` | Which tabs exist and which files each tab documents. |
| `_walkthrough/content/<tab>.html` | Hand-written prose per tab: intro, analogy, pipeline SVG, prerequisites, glossary table, MCQ JSON. |
| `_walkthrough/annotations/<tab>.json` | Per-file purpose, section splits, line notes and critical-line marks. |
| `_walkthrough/template.html` | Page shell: CSS design tokens, all JavaScript, `<!--TABS-->` / `<!--PANELS-->` / `<!--STATS-->` / `<!--MANIFEST-->` placeholders. |
| `_walkthrough/build.py` | Escapes sources, adds line anchors, applies annotations, injects fragments, validates, writes the HTML **and** `_walkthrough/coverage-manifest.json`. |
| `_walkthrough/coverage-manifest.json` | Machine-readable record: every documented file with sha1, line count, sections, notes, question count, glossary size. |

Excluded on purpose: `.venv`, `__pycache__`, `.gradio`. `.env` is **never** opened; only the
names `WATSONX_URL`, `WATSONX_PROJECT_ID`, `WATSONX_APIKEY` may appear.

Build commands (run inside `IBM_GENAI_TRAINING\_walkthrough`):

```powershell
python build.py --diagnose   # list every broken snippet, writes nothing
python build.py --check      # validate everything, writes nothing
python build.py              # validate + write HTML and manifest
```

---

## 2. Update workflow

### A. A new folder appeared (new tab)

1. **Inventory it**: list `.py` files (skip `.venv`, `__pycache__`, `.gradio`) and read each one
   fully. Note the entry point, the data flow, the external services, and any defects.
2. **Register the tab** in `_walkthrough/config.json`:

   ```json
   {
     "id": "myfolder",
     "label": "myfolder",
     "emoji": "\ud83e\uddf1",
     "tagline": "One sentence about what this folder teaches",
     "files": [
       { "path": "myfolder/main.py" },
       { "path": "myfolder/data/sample.json", "lang": "json", "excerpt": [1, 40] }
     ]
   }
   ```

   * `id` must be unique and URL-safe: it becomes the tab, `#hash`, panel id and localStorage key.
   * `lang` defaults to `python`. Set `"highlight": false` for HTML/JS/CSS/bash.
   * `excerpt: [first, last]` documents a slice of a huge file and prints an `excerpt` badge.

3. **Write `annotations/<id>.json`** (schema in section 4). Cover every file listed in config.
4. **Write `content/<id>.html`** (schema in section 5). It must contain one
   `<!-- CODE:<path> -->` marker per configured file - same order as config is recommended - the
   glossary table, and the quiz block with **at least 20** questions.
5. Build, fix every diagnostic, then verify in a browser (section 7).

### B. A new `.py` file appeared in an existing folder

1. Read the file. Decide whether it deserves its own card (always) and where it belongs in the
   reading order - put it in the `files` array at that position. Card ids (`<tab>-f<N>`) shift when
   you insert earlier, so also fix any `href="#<tab>-f<N>"` links in the pipeline SVG.
2. Add an entry to `annotations/<tab>.json` under `files` with `purpose`, optional `run`, `meta`,
   `sections`, `notes`.
3. Add `<!-- CODE:<path> -->` to `content/<tab>.html` at the point where the reader should meet it.
4. Add the new file's acronyms to that tab's glossary and at least one MCQ about it (the tab total
   must stay ≥ 20; add more when a whole new concept appears).
5. Build + verify.

### C. Existing code changed

1. `python build.py --diagnose` - every `snippet not found` / `matches N lines` line names the file
   and note. Fix the snippet or add `"occ"`.
2. Read the diff. If behaviour changed, update the matching `text` in the note - a stale explanation
   is worse than no explanation.
3. Rebuild. The manifest sha1 values refresh automatically.

### D. A file was deleted or moved

Remove it from `config.json`, from `annotations/<tab>.json`, and its `<!-- CODE:... -->` marker.
If the tab now documents nothing, remove the tab and its fragment.

---

## 3. Non-negotiable content rules

* **Audience**: a Class-11 student with no AI background. Short sentences, concrete verbs, no
  unexplained jargon in the first mention.
* **Indian context**: use kirana shop stock registers, IRCTC enquiry counters, Zomato/Swiggy
  re-ranking, UPI, coaching-class notice boards, hostel mess menus. Analogies must clarify the
  mechanism, not decorate it.
* **Explain the code, not the topic.** Every note names the actual line behaviour and, when useful,
  the failure it prevents.
* **Mark real defects**. Use `"kind": "watch"` for genuine bugs, `"kind": "crit"` for concepts that
  break everything if misunderstood, `"kind": "info"` for everything else. Never invent a bug.
* **No dates, no "recently", no time estimates.**
* **No secrets.** Only variable names. Never read or quote `.env`, tokens or API keys.
* **Consistency**: keep the tone, heading shapes and component classes already used in the other
  tabs. New CSS belongs in `template.html` only if no existing class fits.

---

## 4. `annotations/<tab>.json` schema

```json
{
  "folder": "myfolder",
  "files": {
    "myfolder/main.py": {
      "purpose": "One paragraph: what the file is for and why it exists.",
      "run": "python main.py --flag",
      "meta": ["121 lines", "argparse", "chat loop"],
      "sections": [
        { "title": "Imports", "start": "import os", "why": "What this block establishes." },
        { "title": "The loop", "start": "def main():", "occ": 1, "why": "..." }
      ],
      "notes": [
        { "at": "client = chromadb.Client()", "lines": 1, "title": "In-memory client",
          "text": "Plain text. `code` and **bold** are allowed.", "kind": "info" },
        { "at": "except Exception as e:", "occ": 2, "lines": 2, "title": "Swallowed error",
          "text": "...", "kind": "watch" }
      ]
    }
  }
}
```

Rules the builder enforces:

* **Snippet anchoring.** `start` / `at` is a substring of one source line. It must match exactly
  one line; if the same text appears more than once, add `"occ": N` (1-based).
* **Sections.** `start` values must be unique per file and are sorted automatically. The first
  section always begins at line 1 (a header block before the first snippet is included). Each
  section ends on the line before the next section starts; the last runs to EOF. Sections must not
  overlap or be empty. Every line of the file is therefore shown exactly once.
* **Excerpts.** With `"excerpt": [a, b]` in config, snippet line numbers refer to the excerpt, not
  the full file, and any snippet outside the excerpt fails the build.
* **Notes.** `lines` is how many lines from `at` the note covers (default 1). `kind` is one of
  `info`, `crit`, `watch`. Notes are numbered automatically and pinned on their first line.
* **Empty files.** A 0-byte file needs only `purpose`, `why` and `meta`; the builder renders an
  "empty file" card.

Common diagnostics:

| Message | Fix |
|---|---|
| `snippet not found` | The text changed or you copied it from a docstring into code. Re-read the file. |
| `matches N lines ...` | Use a longer snippet or add `"occ"`. |
| `occ N out of range` | The file has fewer occurrences than you assumed. |
| `two notes start on line N` | Two notes anchored to the same source line; merge the text or move one anchor to a different line. |
| `section ... is empty` | Two sections start on the same line - remove one. |

---

## 5. `content/<tab>.html` schema

Required pieces, in this order:

1. `<section class="card">` - **In one breath**: 2-3 sentences plus a short list.
2. `<section class="card analogy">` - Indian analogy that maps to the real components.
3. `<section class="card">` - **Pipeline**: inline `<svg class="pipeline">` with
   `<a class="node" href="#<tab>-f<N>">` boxes and `<path class="arrow">` arrows. Include
   `<defs><marker id="head" .../></defs>` because the CSS uses `marker-end:url(#head)`.
4. `<section class="card">` - **Before you run it**: install commands, env var names, expected
   output, known pitfalls.
5. One `<!-- CODE:<path> -->` marker per configured file, in reading order.
6. `<section class="card glossary-card" id="glossary-<tab>">` with a
   `<div class="glossary-filter"><input type="search" ...></div>` and a table whose body is
   `<tbody class="glossary-body">`, one `<tr class="gtr">` per term:
   `<td>ACRONYM<span>Full Form</span></td><td>Full Form</td><td>What it means here</td>`.
   Every acronym actually used in the tab must appear; do not list terms the reader will not meet.
7. `<section class="card quiz-card" id="quiz-<tab>">`:

   ```html
   <div class="quiz-head">
     <div><h3>Check your understanding - 24 MCQs</h3><p class="small">...</p></div>
     <div class="quiz-score"><b>0 / 24</b><span class="muted">0 answered</span></div>
   </div>
   <div class="qbar"><i></i></div>
   <div class="qcontrols">
     <button class="qchip" type="button" data-filter="all" aria-pressed="true">All</button>
     <button class="qchip" type="button" data-filter="unanswered" aria-pressed="false">Unanswered</button>
     <button class="qchip" type="button" data-filter="wrong" aria-pressed="false">Wrong</button>
     <button class="qchip" type="button" data-filter="easy" aria-pressed="false">Easy</button>
     <button class="qchip" type="button" data-filter="medium" aria-pressed="false">Medium</button>
     <button class="qchip" type="button" data-filter="hard" aria-pressed="false">Hard</button>
     <button class="qchip" type="button" data-action="shuffle">Shuffle</button>
     <button class="qchip" type="button" data-action="retry">Retry wrong only</button>
     <button class="qchip" type="button" data-action="reset">Reset</button>
   </div>
   <div class="quiz-list"></div>
   <script type="application/json" class="quiz">[ ...questions... ]</script>
   ```

   Question object: `{"q": "...", "opts": ["A","B","C","D"], "a": 0, "why": "...",
   "topic": "short-tag", "diff": "easy|medium|hard"}`. Exactly four options, `a` is the 0-based
   index of the correct one, `why` explains why the right answer is right **and** why the tempting
   wrong one fails. Keep at least one `hard` question per tab and mix file-level facts with concepts.

   The static text `<b>0 / N</b>` and the heading `- N MCQs` must match the real count; the builder
   validates the count but not this label, so update both.

JS hook notes: the script reads `.quiz-data` blocks, so keep
`<script type="application/json" class="quiz">` for authoring; the builder converts it into a
`quiz-data` block. Do not put the questions anywhere else.

---

## 6. What the JavaScript gives you (do not rename these)

| Feature | Hook |
|---|---|
| Folder tabs | `.tab[data-tab]`, `#panel-<id>`, `aria-selected`, `#hash` |
| Note pins | `.pin[data-note="<fid>-note<n>"]`, `.note[data-target][data-range][data-file]` |
| Highlight state | `.cl.hl` on highlighted source lines |
| Search | `#search`, filters `.filecard`, `.gtr`, `.qcard`, generic `.card` |
| Copy buttons | `.code-wrap` + `pre.code > code > .cl` (buttons are auto-created) |
| Theme | `data-theme` on `<html>`, stored in `wt-theme` |
| Active tab | `wt-tab`; quiz state in `wt-quiz-v1:<tab>`; progress in `wt-ring:<tab>` |
| Print | `@media print` shows every panel |

---

## 7. Definition of done

Run all of these; do not report the task complete otherwise.

```powershell
cd IBM_GENAI_TRAINING\_walkthrough
python build.py --diagnose     # must print "no snippet problems"
python build.py                # must print the per-tab summary
cd ..
Select-String -Path code-walkthrough.html -Pattern 'api_key="[A-Za-z0-9_-]{20,}"','Bearer [A-Za-z0-9]{20,}'   # must be empty
```

Then open `code-walkthrough.html` in a browser (or `file:///C:/.../code-walkthrough.html`) and check:

1. Every tab opens, has a non-trivial height, and shows no console error.
2. Inside each file card: `.cl` count equals the real line count (or the excerpt length), and there
   are **no duplicate ids**.
3. Clicking a pin highlights the expected lines and flashes the matching note; clicking a note does
   the reverse.
4. Answering a question turns the correct option green, the chosen wrong option rose, shows the
   `why` text, updates `x / N` and the ring; reloading keeps the state.
5. `Retry wrong only`, `Reset`, the difficulty filters and `Shuffle` all work.
6. Search narrows the current tab and shows the empty state for nonsense input.
7. Theme toggle flips `data-theme` and survives a reload; `#<tab>` deep link opens that tab.
8. Every tab has ≥ 20 questions and a glossary with no placeholder rows.

Automation used previously (Playwright, adapted): count `.qcard`, `.gtr`, `.cl`, duplicate ids per
card; click `.opt`; read `localStorage['wt-quiz-v1:<tab>']` after reload.

---

## 8. Known defects already documented (keep them, and keep the notes honest)

* `Gradio/qbot.py` - duplicated `retriever()`; `TRUNCATE_INPUT_TOKENS: 3`; `share=True` with `0.0.0.0`.
* `icebreaker/modules/llm_interface.py` - `truncate_input_tokens=3`; unnecessary `global config`.
* `icebreaker/modules/query_engine.py` - retrieval computed then discarded; string returned on the
  error path while `main.py` expects an object.
* `genai_flask_app/static/script.js` - `modelSelect.value = 'llama3'` matches no option; model
  output rendered with `innerHTML`.
* `genai_flask_app/llm_test.py` - prints `.content` on a dict returned by `model_dump()`.
* `Code/langchain_context_retreival.py` - `watsonx_embedding()` is undefined in that file.
* `Code/RAG_BASIC.py` - `return_message=True` should be `return_messages`; `qa` shadows the function name.
* `Code/LANGCHAIN_AGENT_AI.py` - `eval()` in a tool; 100-token cap too small for a ReAct trace.
* `Food_Recommender/interactive_search.py` - percentage can be negative; wildcard import.
* `Code/data_extraction.py` - empty placeholder.

When you fix one of these, update its note: change `kind` from `watch` to `info` only if the code
really changed, and edit the text to describe the fixed behaviour.

---

## 9. Adding a fourth/fifth model, asset or tab later - checklist

- [ ] `config.json` tab entry with unique `id`, `label`, `emoji`, `tagline`, `files`
- [ ] `annotations/<id>.json` covering every file (purpose + sections + notes + watch-outs)
- [ ] `content/<id>.html` with the six required sections, `<!-- CODE:... -->` per file, glossary, ≥ 20 MCQs
- [ ] pipeline SVG links point at the correct `<tab>-f<N>` ids (recount after inserting files)
- [ ] static MCQ label text matches the real count
- [ ] `python build.py` clean, manifest updated, browser checks from section 7 pass
- [ ] no `.env` value anywhere in the HTML
