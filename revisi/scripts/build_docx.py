"""Render the revised manuscript markdown into an IJAAS-formatted .docx.

Formatting follows the submitted version of this paper, "[1] paper judol ijaas.docx":
A4 single column, justified body with a half-inch first-line indent, Heading 1 for the
numbered sections, bold body text for the unnumbered back matter, centered captions
above tables and below figures, three rules per data table, and 8pt references on a
hanging indent. Every [n] in the running text is an internal hyperlink to entry n of
the reference list.

Reads revisi/manuscript/*.md in filename order and substitutes {{token}} values from
revisi/out/tokens_filled.json. Unresolved tokens stay visible as [[MISSING: name]] and
are listed on stderr, so a half-filled draft can never be mistaken for a finished one.
"""
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image

ROOT = Path("/home/ftib/ultralytics")
MS = ROOT / "revisi/manuscript"
OUT = ROOT / "revisi/out"
TEMPLATE = ROOT / "revisi/ijaas_gfa.docx"
FIGS = ROOT / "revisi/figures"

BOLD = re.compile(r"\*\*(.+?)\*\*")
CITE = re.compile(r"\[(\d+)\]")
TOKEN = re.compile(r"\{\{([^}]+)\}\}")
FIGREF = re.compile(r"^\[\[FIGURE:([^\]|]+)\|(.*)\]\]$")
INCLUDE = re.compile(r"^\[\[INCLUDE:([^\]]+)\]\]$")
TABLE_CAPTION = re.compile(r"^\*\*Table \d+\.")
REF_ENTRY = re.compile(r"^\[(\d+)\]\s")
LIST_ITEM = re.compile(r"^(\d+\.|[-*])\s+")
# The back matter carries no section number, so the submitted version sets these as
# bold body text instead of headings.
BACKMATTER = {"ACKNOWLEDGMENTS", "FUNDING INFORMATION", "AUTHOR CONTRIBUTIONS STATEMENT",
              "CONFLICT OF INTEREST STATEMENT", "DATA AVAILABILITY", "ETHICS STATEMENT",
              "REFERENCES", "BIOGRAPHIES OF AUTHORS"}
BODY_INDENT = Inches(0.5)
HANG = Inches(0.39)


def load_tokens():
    f = OUT / "tokens_filled.json"
    return json.loads(f.read_text()) if f.exists() else {}


def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(f"w:{k}"), v)
    return e


def blank_body(doc):
    body = doc.element.body
    for child in list(body):
        if not child.tag.endswith("}sectPr"):
            body.remove(child)
    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.27), Inches(11.69)
    s.left_margin, s.right_margin = Inches(1.18), Inches(0.98)
    s.top_margin, s.bottom_margin = Inches(0.98), Inches(0.98)
    return doc


def add_runs(p, text, link=True):
    """**bold** spans become bold runs, and [12] becomes a link to reference 12."""
    for i, part in enumerate(BOLD.split(text)):
        if not part:
            continue
        bold = bool(i % 2)
        for j, piece in enumerate(CITE.split(part) if link else [part]):
            if not piece:
                continue
            if j % 2:
                run = p.add_run(f"[{piece}]")
                h = el("w:hyperlink", anchor=f"ref_{piece}")
                run._r.addprevious(h)
                h.append(run._r)
            else:
                run = p.add_run(piece)
            run.bold = bold


def emit(doc, text, style="Normal", align=AL.JUSTIFY, size=None, bold=False,
         first_line=None, left=None, space_after=None, link=True):
    p = doc.add_paragraph(style=style)
    add_runs(p, text, link)
    pf = p.paragraph_format
    pf.alignment = align
    pf.first_line_indent = first_line
    pf.left_indent = left
    if space_after is not None:
        pf.space_after = space_after
    for r in p.runs:
        if size:
            r.font.size = size
        if bold:
            r.bold = True
    return p


def set_borders(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(old)
    b = el("w:tcBorders")
    for edge, val in edges.items():
        b.append(el(f"w:{edge}", val=val, sz="4", space="0", color="auto"))
    tcPr.append(b)


def fill(cell, lines, size=Pt(8), align=AL.LEFT, link=False):
    cell.text = ""
    for n, line in enumerate(lines):
        p = cell.paragraphs[0] if n == 0 else cell.add_paragraph()
        add_runs(p, line, link)
        p.paragraph_format.alignment = align
        p.paragraph_format.space_after = Pt(0)
        for r in p.runs:
            r.font.size = size


def add_figure(doc, path):
    """Full text width, scaled down when that would make the figure taller than half
    the text block. The panels range from square to 5:1, so a fixed width left the wide
    ones small and the tall ones running over a page."""
    sec = doc.sections[0]
    avail = sec.page_width - sec.left_margin - sec.right_margin
    cap = int((sec.page_height - sec.top_margin - sec.bottom_margin) * 0.45)
    w, h = Image.open(path).size
    doc.add_picture(str(path), width=avail if avail * h / w <= cap else int(cap * w / h))
    doc.paragraphs[-1].alignment = AL.CENTER


def add_table(doc, rows, data=True):
    """Data tables take the submitted version's three rules: one above the header, one
    below it, one under the last row. Layout tables keep a full grid."""
    t = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
    t.style = "Normal Table" if data else "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        for ci, txt in enumerate(row):
            cell = t.cell(ri, ci)
            fill(cell, [txt], Pt(8), AL.CENTER if ri == 0 or ci else AL.LEFT)
            if ri == 0:
                for r in cell.paragraphs[0].runs:
                    r.bold = True
            if data and (ri == 0 or ri == len(rows) - 1):
                set_borders(cell, **({"top": "single", "bottom": "single"} if ri == 0
                                     else {"bottom": "single"}))
    doc.add_paragraph()


def make_sub(missing):
    def sub(m):
        key = m.group(1).strip()
        val = load_tokens().get(key)
        if val is None:
            missing.add(key)
            return f"[[MISSING: {key}]]"
        return str(val)
    return sub


def front(doc, text, sub):
    """Title block plus the article-info box that holds the abstract and keywords."""
    parts = [TOKEN.sub(sub, b.replace("\n", " ").strip()) for b in text.split("\n\n") if b.strip()]
    emit(doc, parts[0].lstrip("# ").strip(), style="Title", align=AL.CENTER, size=Pt(16), link=False)
    emit(doc, parts[1], align=AL.CENTER, bold=True, link=False)
    for aff in parts[2:4]:
        emit(doc, aff, align=AL.CENTER, size=Pt(8), space_after=Pt(0), link=False)
    doc.add_paragraph()

    i = parts.index("**Corresponding Author:**")
    corr = parts[i:parts.index("## ABSTRACT")]
    abstract = [p for p in parts[parts.index("## ABSTRACT") + 1:] if not p.startswith("**Keywords:**")]
    keywords = next(p for p in parts if p.startswith("**Keywords:**"))

    t = doc.add_table(rows=5, cols=3)
    t.style = "Normal Table"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.cell(1, 2).merge(t.cell(2, 2))
    t.cell(2, 0).merge(t.cell(3, 0))
    t.cell(4, 0).merge(t.cell(4, 2))
    for row in t.rows:
        for cell, w in zip(row.cells, (Inches(1.95), Inches(0.2), Inches(4.04))):
            cell.width = w

    fill(t.cell(0, 0), ["**Article Info**"])
    fill(t.cell(0, 2), ["**ABSTRACT**"])
    fill(t.cell(1, 0), ["**Article history:**", "Received month dd, yyyy",
                        "Revised month dd, yyyy", "Accepted month dd, yyyy"])
    fill(t.cell(2, 0), ["**Keywords:**"] + [k.strip() for k in
                                            keywords.split("**Keywords:**")[1].split(";")])
    fill(t.cell(1, 2), abstract, Pt(9), AL.JUSTIFY, link=True)
    fill(t.cell(3, 2), ["This is an open access article under the CC BY-SA license."],
         Pt(8), AL.CENTER)
    fill(t.cell(4, 0), corr)

    for c in t.rows[0].cells:
        set_borders(c, top="double", bottom="single" if c is not t.cell(0, 1) else "nil")
    set_borders(t.cell(1, 0), top="single", bottom="single")
    set_borders(t.cell(2, 0), top="single", bottom="single")
    set_borders(t.cell(3, 2), bottom="single")
    set_borders(t.cell(4, 0), bottom="double")
    doc.add_paragraph()


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(doc, text, missing, ctx):
    sub = make_sub(missing)

    def body(txt):
        """Running text, and the caption that sits above the table it announces."""
        if TABLE_CAPTION.match(txt):
            emit(doc, txt, align=AL.CENTER, space_after=Pt(3))
        elif ctx["section"] == "BIOGRAPHIES OF AUTHORS":
            emit(doc, txt, size=Pt(8))
        else:
            emit(doc, txt, first_line=BODY_INDENT)

    lines = text.split("\n")
    i, para = 0, []

    def flush():
        if para:
            body(TOKEN.sub(sub, " ".join(para)))
            para.clear()

    while i < len(lines):
        s = lines[i].strip()

        if not s:
            flush()
            i += 1
            continue

        inc = INCLUDE.match(s)
        if inc:
            flush()
            src = OUT / inc.group(1).strip()
            if src.exists():
                render(doc, src.read_text(), missing, ctx)
            else:
                missing.add(f"include:{inc.group(1).strip()}")
                emit(doc, f"[[MISSING TABLE: {inc.group(1).strip()}]]")
            i += 1
            continue

        fig = FIGREF.match(s)
        if fig:
            flush()
            name, cap = fig.group(1).strip(), fig.group(2).strip()
            path = FIGS / name
            if path.exists():
                add_figure(doc, path)
            else:
                missing.add(f"figure:{name}")
                emit(doc, f"[[MISSING FIGURE: {name}]]", align=AL.CENTER)
            emit(doc, TOKEN.sub(sub, cap), align=AL.CENTER, space_after=Pt(10))
            i += 1
            continue

        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            head = TOKEN.sub(sub, s.lstrip("# ").strip())
            ctx["section"] = head if level == 2 else ctx["section"]
            if head in BACKMATTER:
                emit(doc, head, bold=True, link=False)
            else:
                emit(doc, head, style=f"Heading {min(level - 1, 3)}", align=AL.LEFT, link=False)
            i += 1
            continue

        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = split_row(lines[i])
                if not all(set(c) <= set("-: ") for c in r):
                    rows.append([TOKEN.sub(sub, c) for c in r])
                i += 1
            if rows:
                add_table(doc, rows, data=ctx["numbered_table"])
                ctx["numbered_table"] = False
            continue

        ref = REF_ENTRY.match(s)
        if ref:
            # IEEE entries are one line each with no blank line between them, so they
            # break the paragraph themselves or the list renders as one block. Each one
            # carries the bookmark that the [n] citations in the body point at.
            flush()
            p = emit(doc, TOKEN.sub(sub, s), size=Pt(8), first_line=-HANG, left=HANG,
                     space_after=Pt(0), link=False)
            ctx["bookmark"] += 1
            p._p.insert(0, el("w:bookmarkStart", id=str(ctx["bookmark"]),
                              name=f"ref_{ref.group(1)}"))
            p._p.append(el("w:bookmarkEnd", id=str(ctx["bookmark"])))
            i += 1
            continue

        if LIST_ITEM.match(s):
            flush()
            emit(doc, TOKEN.sub(sub, s), left=Inches(0.25), first_line=Inches(-0.25),
                 space_after=Pt(0))
            i += 1
            continue

        if TABLE_CAPTION.match(s):
            ctx["numbered_table"] = True
        para.append(s)
        i += 1

    flush()


def main():
    doc = blank_body(Document(str(TEMPLATE)))
    missing = set()
    ctx = {"section": "", "bookmark": 0, "numbered_table": False}
    files = sorted(MS.glob("*.md"))
    if not files:
        sys.exit("no manuscript sections found")
    for f in files:
        if f.name.startswith("01_"):
            front(doc, f.read_text(), make_sub(missing))
        else:
            render(doc, f.read_text(), missing, ctx)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "manuscript_ijaas.docx"
    doc.save(str(dest))
    print(f"wrote {dest} from {len(files)} sections")
    if missing:
        print(f"\n{len(missing)} unresolved placeholders:", file=sys.stderr)
        for k in sorted(missing):
            print(f"  {k}", file=sys.stderr)


if __name__ == "__main__":
    main()
