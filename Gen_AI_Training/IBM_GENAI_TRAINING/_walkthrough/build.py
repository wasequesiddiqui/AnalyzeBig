#!/usr/bin/env python3
"""Assemble ``code-walkthrough.html`` from the content fragments and Python sources.

The kit is deliberately split so the HTML can be regenerated instead of hand-edited:

    config.json                  which tabs exist and which files each tab documents
    content/<tab>.html           prose: intro, pipeline, prerequisites, glossary, quiz JSON
    annotations/<tab>.json       per-file purpose plus line notes and critical-line marks
    template.html                page shell: CSS, JavaScript, placeholders
    build.py                     this file - escape, anchor, inject, validate, write manifest

Usage:
    python build.py            regenerate the HTML and the manifest
    python build.py --check    validate only, write nothing

Every note in the annotation files is anchored by a *code snippet* rather than a line
number, so the notes keep working when code is inserted above them.  A snippet must
match exactly one line unless ``occ`` (1-based) is given.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parent
CONFIG = json.loads((KIT / "config.json").read_text(encoding="utf-8"))
TEMPLATE = KIT / "template.html"
MIN_QUESTIONS = 20
VALID_KINDS = {"info", "crit", "watch"}
SOFT = False
ERRORS: list[str] = []


class BuildError(Exception):
    """A problem a human must fix in the kit, reported with file context."""


# --------------------------------------------------------------------------- helpers
def esc(text: str) -> str:
    return html.escape(text, quote=False)


def inline(text: str) -> str:
    """Escape author text, then allow `code` and **bold** inline markup."""
    out = esc(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def find_line(lines: list[str], snippet: str, occ: int | None, where: str) -> int:
    """Return the 0-based index of ``snippet`` inside ``lines``."""
    hits = [i for i, line in enumerate(lines) if snippet in line]
    problem = None
    if not hits:
        problem = f"{where}: snippet not found -> {snippet!r}"
    elif occ is None and len(hits) > 1:
        numbers = ", ".join(str(h + 1) for h in hits[:8])
        problem = (
            f"{where}: snippet matches {len(hits)} lines ({numbers}); "
            f"add an \"occ\" or use a longer snippet -> {snippet!r}"
        )
    elif occ is not None and (occ < 1 or occ > len(hits)):
        problem = f"{where}: occ {occ} out of range (found {len(hits)} matches) -> {snippet!r}"
    if problem:
        if not SOFT:
            raise BuildError(problem)
        ERRORS.append(problem)
        return hits[0] if hits else 0
    return hits[occ - 1] if occ else hits[0]


def read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n") if text else []


# --------------------------------------------------------------------------- annotations
def load_annotations(tab_id: str) -> dict:
    path = KIT / "annotations" / f"{tab_id}.json"
    if not path.exists():
        raise BuildError(f"missing annotation file: {path.relative_to(KIT)}")
    return json.loads(path.read_text(encoding="utf-8")).get("files", {})


def resolve_sections(lines: list[str], entry: dict, where: str) -> list[dict]:
    """Turn the authored section list into concrete [start, end] line ranges."""
    authored = entry.get("sections") or []
    if not authored:
        return [{"title": "Full file", "why": "", "start": 1, "end": len(lines)}]

    marks = []
    for item in authored:
        start = find_line(lines, item["start"], item.get("occ"), f"{where} section {item['title']!r}")
        marks.append({"title": item["title"], "why": item.get("why", ""), "start": start})
    marks.sort(key=lambda m: m["start"])

    sections = []
    for index, mark in enumerate(marks):
        # mark["start"] is 0-based; convert to 1-based line numbers.
        start = 1 if index == 0 else mark["start"] + 1
        end = marks[index + 1]["start"] if index + 1 < len(marks) else len(lines)
        if end < start:
            raise BuildError(f"{where}: section {mark['title']!r} is empty (it starts after the next section)")
        sections.append({"title": mark["title"], "why": mark["why"], "start": start, "end": end})
    return sections


def resolve_notes(lines: list[str], entry: dict, sections: list[dict], fid: str, where: str) -> list[dict]:
    notes = []
    for raw in entry.get("notes") or []:
        kind = raw.get("kind", "info")
        if kind not in VALID_KINDS:
            raise BuildError(f"{where}: unknown note kind {kind!r}")
        start = find_line(lines, raw["at"], raw.get("occ"), f"{where} note {raw.get('title')!r}")
        span = max(1, int(raw.get("lines", 1)))
        end = min(len(lines), start + span)
        notes.append(
            {
                "id": f"{fid}-note{len(notes) + 1}",
                "title": raw.get("title") or "Note",
                "text": raw.get("text", ""),
                "kind": kind,
                "start": start + 1,
                "end": end,
                "fid": fid,
            }
        )
    notes.sort(key=lambda n: n["start"])
    seen_lines: dict[int, str] = {}
    for note in notes:
        if note["start"] in seen_lines:
            problem = (
                f"{where}: two notes start on line {note['start']} "
                f"({seen_lines[note['start']]!r} and {note['title']!r}); merge them or move one anchor"
            )
            if not SOFT:
                raise BuildError(problem)
            ERRORS.append(problem)
        seen_lines[note["start"]] = note["title"]
    for note in notes:
        for section in sections:
            if section["start"] <= note["start"] <= section["end"]:
                note["section"] = section["title"]
                break
    return notes


# --------------------------------------------------------------------------- rendering
def render_notes(notes: list[dict]) -> str:
    if not notes:
        return ""
    items = []
    for note in notes:
        rng = f"line {note['start']}" if note["start"] == note["end"] else f"lines {note['start']}-{note['end']}"
        icon = {"info": "", "crit": "!", "watch": "?"}[note["kind"]]
        items.append(
            f'<li class="note note-{note["kind"]}" id="{note["id"]}" data-target="{note["fid"]}-L{note["start"]}" '
            f'data-range="{note["start"]}-{note["end"]}" data-file="{note["fid"]}">'
            f'<span class="note-n">{icon}</span>'
            f'<div class="note-body"><b>{inline(note["title"])}</b>'
            f'<span class="note-range">{rng}</span>'
            f'<p>{inline(note["text"])}</p></div></li>'
        )
    return '<ul class="notes">' + "".join(items) + "</ul>"


def render_file_card(tab_id: str, index: int, spec: dict, entry: dict) -> tuple[str, dict]:
    rel = spec["path"]
    path = ROOT / rel
    where = f"{tab_id}/{rel}"
    if not path.exists():
        raise BuildError(f"{where}: file does not exist")
    lines = read_lines(path)

    excerpt = spec.get("excerpt")
    if excerpt:
        first, last = int(excerpt[0]), int(excerpt[1])
        if first < 1 or last > len(lines) or first > last:
            raise BuildError(f"{where}: excerpt {excerpt} outside 1..{len(lines)}")
        excerpt_lines = lines[first - 1 : last]
        offset = first - 1
    else:
        excerpt_lines = lines
        offset = 0

    working = excerpt_lines
    fid = f"{tab_id}-f{index}"
    if not working:
        card = (
            f'<article class="filecard" id="{fid}" data-file="{fid}" data-path="{esc(rel)}">'
            f'<header class="fc-head"><div><h3><span class="fc-name">{esc(Path(rel).name)}</span>'
            f'<span class="fc-path">{esc(rel)}</span></h3>'
            f'<p class="fc-purpose">{inline(entry.get("purpose", "No purpose written yet."))}</p></div>'
            f'<div class="fc-meta"><span class="badge badge-empty">empty file (0 lines)</span></div></header>'
            f'<div class="sec"><p class="sec-why">{inline(entry.get("why", ""))}</p></div></article>'
        )
        return card, {
            "path": rel,
            "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
            "lines": 0,
            "documented_lines": 0,
            "documented": "0",
            "sections": 0,
            "notes": 0,
        }
    sections = resolve_sections(working, entry, where)
    notes = resolve_notes(working, entry, sections, fid, where)

    pins: dict[int, str] = {}
    numbers: dict[int, float] = {}
    classes: dict[int, str] = {}
    for position, note in enumerate(notes, 1):
        note["number"] = position
    for note in notes:
        pins[note["start"]] = note["id"]
        numbers[note["id"]] = note["number"]
        for number in range(note["start"], note["end"] + 1):
            if note["kind"] == "info":
                classes.setdefault(number, "")
            elif note["kind"] == "watch":
                classes[number] = " ln-watch"
            elif note["kind"] == "crit" and classes.get(number) != " ln-watch":
                classes[number] = " ln-crit"

    rendered_sections = []
    for section in sections:
        section_notes = [n for n in notes if n.get("section") == section["title"]]
        section_pins = {
            line: pid for line, pid in pins.items() if section["start"] <= line <= section["end"]
        }
        rows = []
        for number in range(section["start"], section["end"] + 1):
            text = working[number - 1]
            pin = section_pins.get(number)
            pin_html = f'<span class="pin" data-note="{pin}">{int(numbers[pin])}</span>' if pin else ""
            rows.append(
                f'<span class="cl{classes.get(number, "")}" id="{fid}-L{number}" data-n="{number}">'
                f'{pin_html}{esc(text) or " "}</span>'
            )
        lang = spec.get("lang", "python")
        highlight = spec.get("highlight", True)
        rendered_sections.append(
            '<div class="sec">'
            f'<h4 class="sec-title">{inline(section["title"])}</h4>'
            + (f'<p class="sec-why">{inline(section["why"])}</p>' if section.get("why") else "")
            + f'<div class="code-wrap"><button class="copy" type="button">Copy</button>'
            f'<pre class="code" data-lang="{lang}" data-highlight="{str(highlight).lower()}" '
            f'style="counter-reset: ln {section["start"] - 1}"><code>'
            + "\n".join(rows)
            + "</code></pre></div>"
            + render_notes(section_notes)
            + "</div>"
        )

    badge = ""
    if excerpt:
        badge = f'<span class="badge badge-excerpt">excerpt: lines {excerpt[0]}-{excerpt[1]} of {len(lines)}</span>'
    elif not working:
        badge = '<span class="badge badge-empty">empty file (0 lines)</span>'

    meta = entry.get("meta") or []
    meta_html = "".join(f'<span class="chip">{inline(item)}</span>' for item in meta)

    card = (
        f'<article class="filecard" id="{fid}" data-file="{fid}" data-path="{esc(rel)}">'
        f'<header class="fc-head"><div><h3><span class="fc-name">{esc(Path(rel).name)}</span>'
        f'<span class="fc-path">{esc(rel)}</span></h3>'
        f'<p class="fc-purpose">{inline(entry.get("purpose", "No purpose written yet."))}</p></div>'
        f'<div class="fc-meta">{badge}{meta_html}</div></header>'
        + (f'<p class="fc-run"><span>Run it</span><code>{inline(entry["run"])}</code></p>' if entry.get("run") else "")
        + "".join(rendered_sections)
        + "</article>"
    )

    manifest = {
        "path": rel,
        "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
        "lines": len(lines),
        "documented_lines": len(working),
        "documented": f"{excerpt[0]}-{excerpt[1]}" if excerpt else f"1-{len(lines)}",
        "sections": len(sections),
        "notes": len(notes),
    }
    return card, manifest


# --------------------------------------------------------------------------- content
def load_fragment(tab_id: str) -> str:
    path = KIT / "content" / f"{tab_id}.html"
    if not path.exists():
        raise BuildError(f"missing content fragment: {path.relative_to(KIT)}")
    return path.read_text(encoding="utf-8")


def extract_quiz(fragment: str, tab_id: str) -> tuple[str, list[dict]]:
    pattern = re.compile(r'<script type="application/json" class="quiz">(.*?)</script>', re.S)
    match = pattern.search(fragment)
    if not match:
        raise BuildError(f"{tab_id}: fragment has no <script class=\"quiz\"> block")
    try:
        questions = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise BuildError(f"{tab_id}: quiz JSON is invalid -> {exc}") from exc
    if len(questions) < MIN_QUESTIONS:
        raise BuildError(f"{tab_id}: only {len(questions)} questions, need at least {MIN_QUESTIONS}")
    for position, question in enumerate(questions, 1):
        missing = {"q", "opts", "a", "why"} - question.keys()
        if missing:
            raise BuildError(f"{tab_id} Q{position}: missing keys {sorted(missing)}")
        if len(question["opts"]) != 4:
            raise BuildError(f"{tab_id} Q{position}: needs exactly 4 options")
        if not 0 <= int(question["a"]) <= 3:
            raise BuildError(f"{tab_id} Q{position}: answer index out of range")
        question["id"] = position
        question.setdefault("topic", "general")
        question.setdefault("diff", "medium")
    return pattern.sub("", fragment), questions


def count_glossary_terms(fragment: str, tab_id: str) -> int:
    body = re.search(r'<tbody class="glossary-body">(.*?)</tbody>', fragment, re.S)
    if not body:
        raise BuildError(f"{tab_id}: glossary table with <tbody class=\"glossary-body\"> is required")
    terms = len(re.findall(r"<tr\b", body.group(1)))
    if terms == 0:
        raise BuildError(f"{tab_id}: glossary has no rows")
    return terms


def inject_code_markers(fragment: str, tab_id: str, files: list[dict]) -> tuple[str, list[dict]]:
    annotations = load_annotations(tab_id)
    markers = re.findall(r"<!--\s*CODE:([^>]+?)\s*-->", fragment)
    expected = [spec["path"] for spec in files]
    for path in expected:
        if path not in markers:
            raise BuildError(f"{tab_id}: fragment lacks a <!-- CODE:{path} --> marker")
    for marker in markers:
        if marker not in expected:
            raise BuildError(f"{tab_id}: fragment has a marker for an unlisted file -> {marker}")

    manifest_files = []
    for index, spec in enumerate(files, 1):
        rel = spec["path"]
        if rel not in annotations:
            raise BuildError(f"{tab_id}: no annotation entry for {rel}")
        card, info = render_file_card(tab_id, index, spec, annotations[rel])
        fragment = re.sub(r"<!--\s*CODE:" + re.escape(rel) + r"\s*-->", lambda _m, c=card: c, fragment, count=1)
        manifest_files.append(info)
    return fragment, manifest_files


def build_tab(tab: dict) -> tuple[str, dict]:
    tab_id = tab["id"]
    fragment = load_fragment(tab_id)
    glossary_terms = count_glossary_terms(fragment, tab_id)
    fragment, files = inject_code_markers(fragment, tab_id, tab["files"])
    fragment, questions = extract_quiz(fragment, tab_id)

    panel = (
        f'<section class="panel" id="panel-{tab_id}" role="tabpanel" aria-labelledby="tab-{tab_id}" hidden>'
        f'<div class="panel-head"><div><h2>{tab["emoji"]} {esc(tab["label"])}</h2>'
        f'<p class="tagline">{inline(tab["tagline"])}</p></div>'
        f'<div class="panel-stats">'
        f'<span class="stat"><b>{len(tab["files"])}</b> files</span>'
        f'<span class="stat"><b>{sum(f["documented_lines"] for f in files):,}</b> lines shown</span>'
        f'<span class="stat"><b>{len(questions)}</b> MCQs</span>'
        f'<span class="stat"><b>{glossary_terms}</b> glossary terms</span>'
        f'<span class="ring" data-quiz-tab="{tab_id}"><span class="ring-text">0%</span></span>'
        f"</div></div>"
        f"{fragment}"
        f'<script type="application/json" class="quiz-data" data-tab="{tab_id}">'
        f"{json.dumps(questions, ensure_ascii=False)}</script>"
        f"</section>"
    )
    manifest = {
        "id": tab_id,
        "label": tab["label"],
        "tagline": tab["tagline"],
        "questions": len(questions),
        "glossary_terms": glossary_terms,
        "files": files,
    }
    return panel, manifest


def main() -> int:
    global SOFT
    check_only = "--check" in sys.argv
    diagnose = "--diagnose" in sys.argv
    if diagnose:
        SOFT = True
    template = TEMPLATE.read_text(encoding="utf-8")

    tabs_html = []
    panels = []
    manifest_tabs = []
    total_files = 0
    total_lines = 0
    total_questions = 0

    for index, tab in enumerate(CONFIG["tabs"]):
        try:
            panel, manifest = build_tab(tab)
        except BuildError as exc:
            if diagnose:
                ERRORS.append(f"[{tab['id']}] {exc}")
                continue
            raise
        panels.append(panel)
        manifest_tabs.append(manifest)
        total_files += len(manifest["files"])
        total_lines += sum(f["documented_lines"] for f in manifest["files"])
        total_questions += manifest["questions"]
        active = "true" if index == 0 else "false"
        tabs_html.append(
            f'<button class="tab" role="tab" id="tab-{tab["id"]}" data-tab="{tab["id"]}" '
            f'aria-controls="panel-{tab["id"]}" aria-selected="{active}" tabindex="{0 if index == 0 else -1}">'
            f'<span class="tab-emoji">{tab["emoji"]}</span><span class="tab-label">{esc(tab["label"])}</span>'
            f'<span class="tab-dot" aria-hidden="true"></span></button>'
        )

    stats = (
        f'<span class="chip">{len(CONFIG["tabs"])} folders</span>'
        f'<span class="chip">{total_files} files</span>'
        f'<span class="chip">{total_lines:,} lines</span>'
        f'<span class="chip">{total_questions} MCQs</span>'
    )

    manifest = {
        "kit": "_walkthrough",
        "root": ".",
        "min_questions_per_tab": MIN_QUESTIONS,
        "totals": {
            "tabs": len(CONFIG["tabs"]),
            "files": total_files,
            "lines": total_lines,
            "questions": total_questions,
        },
        "tabs": manifest_tabs,
    }

    page = (
        template.replace("<!--TABS-->", "\n".join(tabs_html))
        .replace("<!--PANELS-->", "\n".join(panels))
        .replace("<!--STATS-->", stats)
        .replace("<!--MANIFEST-->", json.dumps(manifest, indent=2, ensure_ascii=False))
        .replace("{{TITLE}}", esc(CONFIG["title"]))
        .replace("{{SUBTITLE}}", esc(CONFIG["subtitle"]))
    )

    if diagnose:
        if ERRORS:
            print(f"{len(ERRORS)} snippet problem(s):", file=sys.stderr)
            for problem in ERRORS:
                print("  - " + problem, file=sys.stderr)
            return 1
        print("no snippet problems")
        return 0

    if check_only:
        print(f"OK  {len(CONFIG['tabs'])} tabs, {total_files} files, {total_lines} lines, {total_questions} questions")
        return 0

    out = ROOT / CONFIG["output"]
    out.write_text(page, encoding="utf-8")
    manifest_path = KIT / "coverage-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    for tab in manifest_tabs:
        print(f"  {tab['id']:<12} {len(tab['files'])} files  {tab['questions']} MCQs  {tab['glossary_terms']} terms")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"BUILD ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
