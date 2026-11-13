"""Markdown copies of the tables.

`write(name)` turns ``tables/<name>.tex`` into ``tables/<name>.md``: the tabular as a Markdown table,
then the caption. The LaTeX file stays the reference; the copy is for reading and for pasting into
notes. It handles the constructs the generated tables use and nothing more:

* column groups (``\\multicolumn`` in the header) are folded into the column names, "group: column";
* a full-width ``\\multicolumn`` row in the body heads a block and becomes a bold row;
* a stacked cell (a nested ``tabular``) is written on one line;
* rules and spacing are dropped, since a Markdown table has one header rule only.
"""
from __future__ import annotations

import re
from pathlib import Path

import common as C

BEGIN, END = r"\begin{tabular}", r"\end{tabular}"
SYMBOLS = [("pm", "±"), ("downarrow", "↓"), ("uparrow", "↑"), ("times", "×"), ("Delta", "Δ"), ("rho", "ρ"),
           ("lambda", "λ"), ("Sigma", "Σ"), ("dagger", "†"), ("ast", "∗"), ("top", "T"), ("geq", "≥"), ("ge", "≥"),
           ("leq", "≤"), ("le", "≤"), ("cdot", "·"), ("ldots", "…")]
SIZES = ("scriptsize", "footnotesize", "small", "normalsize", "centering", "quad", "qquad")


def _group(s: str, i: int) -> int:
    """Index just after the brace group that opens at ``s[i]``."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def _split(s: str, sep: str) -> list:
    """Split at `sep` outside braces; an escaped separator (``\\&``) does not split."""
    out, depth, last, i = [], 0, 0, 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        depth += (s[i] == "{") - (s[i] == "}")
        if depth == 0 and s.startswith(sep, i):
            out.append(s[last:i])
            last = i + len(sep)
        i += 1
    return out + [s[last:]]


def _rows(body: str) -> list:
    """The rows of a tabular body: split at ``\\\\`` outside braces and outside nested tabulars."""
    out, depth, nested, last, i = [], 0, 0, 0, 0
    while i < len(body):
        if body.startswith(BEGIN, i):
            nested += 1
        elif body.startswith(END, i):
            nested -= 1
        if body.startswith("\\\\", i) and depth == 0 and nested == 0:
            out.append(body[last:i])
            i += 2
            m = re.match(r"\[[^\]]*\]", body[i:])                # the optional row spacing, \\[1pt]
            i += m.end() if m else 0
            last = i
            continue
        if body[i] == "\\":
            i += 2
            continue
        depth += (body[i] == "{") - (body[i] == "}")
        i += 1
    return out + [body[last:]]


def plain(tex: str) -> str:
    """A table cell or a caption as Markdown text."""
    s = tex
    while BEGIN in s:                                            # a stacked cell: its lines side by side
        a = s.index(BEGIN)
        b = s.index(END, a)
        j = a + len(BEGIN)
        if s[j] == "[":
            j = s.index("]", j) + 1
        inner = s[_group(s, j):b]
        s = s[:a] + re.sub(r"\\\\(\[[^\]]*\])?", " ", inner) + s[b + len(END):]
    for cmd, mark in (("textbf", "**"), ("textit", "*"), ("emph", "*"), ("texttt", "`")):
        while f"\\{cmd}{{" in s:
            a = s.index(f"\\{cmd}{{")
            b = _group(s, a + len(cmd) + 1)
            s = s[:a] + mark + s[a + len(cmd) + 2:b - 1].strip() + mark + s[b:]
    s = re.sub(r"\\ref\{([^}]*)\}", r"`\1`", s)
    s = re.sub(r"\\(?:mathrm|textrm|text|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
    for cmd in SIZES:
        s = re.sub(r"\\" + cmd + r"(?![A-Za-z])", " ", s)
    for cmd, sym in SYMBOLS:
        s = re.sub(r"\\" + cmd + r"(?![A-Za-z])", sym, s)
    for old, new in (("{=}", "="), ("\\,", ""), ("\\ ", " "), ("~", " "), ("\\%", "%"), ("\\&", "&"), ("\\#", "#"),
                     ("\\_", "_"), ("$-$", "−"), ("---", "—"), ("--", "–")):
        s = s.replace(old, new)
    s = re.sub(r"\^\{?2\}?", "²", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"^\1", s)
    for old, new in (("^†", "†"), ("^∗", "∗"), ("^*", "∗"), ("^T", "ᵀ"), ("$", ""), ("{", ""), ("}", "")):
        s = s.replace(old, new)
    s = re.sub(r"\s*±\s*", " ± ", s)
    s = re.sub(r"(?<=[0-9])\[", " [", s)                         # a value and its interval
    return re.sub(r"\s+", " ", s).replace("|", "\\|").replace("∗", "\\*").strip()


def _tabular(tex: str, start: int) -> tuple:
    """``(markdown lines, index after the tabular)`` for the tabular that begins at ``tex[start]``."""
    depth, i = 0, start
    while True:                                                  # the END that closes this tabular
        a, b = tex.find(BEGIN, i), tex.find(END, i)
        if a != -1 and a < b:
            depth, i = depth + 1, a + 1
        else:
            depth, i = depth - 1, b + 1
            if depth == 0:
                break
    body = tex[_group(tex, tex.index("{", start + len(BEGIN))):b]
    body = re.sub(r"\\(cmidrule(\([^)]*\))?\{[^}]*\}|addlinespace(\[[^\]]*\])?|toprule|bottomrule)", "", body)
    header, rows, ruled = [], [], False
    for chunk in _rows(body):
        if "\\midrule" in chunk:
            chunk, ruled = re.sub(r"\\midrule(\[[^\]]*\])?", "", chunk), True
        if not chunk.strip():
            continue
        cells, spans = [], []
        for c in _split(chunk.strip(), "&"):
            m = re.match(r"\s*\\multicolumn\{(\d+)\}", c)
            if m:
                k = int(m.group(1))
                j = _group(c, c.index("{", m.end()))             # past the column spec
                cells += [plain(c[j + 1:_group(c, j) - 1])] + [""] * (k - 1)
                spans += [k] + [0] * (k - 1)
            else:
                cells.append(plain(c))
                spans.append(1)
        (rows if ruled else header).append((cells, spans))
    width = max(len(c) for c, _ in header + rows)
    names, grouped = [""] * width, [False] * width
    for cells, spans in header:                                  # fold the header rows into one name per column
        label = ""
        for j in range(min(width, len(cells))):
            label = cells[j] if spans[j] else label
            part = label if (spans[j] != 1 or cells[j]) else ""
            if names[j] and part:                                # "group: column"; a name broken over two rows is joined
                names[j] += (": " if grouped[j] else "" if names[j].endswith("-") else " ") + part
            else:
                names[j] = names[j] or part
            grouped[j] = spans[j] != 1
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * width]
    for cells, spans in rows:
        if spans and spans[0] == width:                          # a full-width row heads a block
            cells = [cells[0] if cells[0].startswith("*") else f"**{cells[0]}**"] + [""] * (width - 1)
        lines.append("| " + " | ".join(cells + [""] * (width - len(cells))) + " |")
    return lines, b + len(END)


def convert(tex: str) -> tuple:
    """``(markdown lines, caption)`` of one generated table file: its tabulars in order, each as a Markdown
    table, with the text between them (the panel titles of a multi-panel table) as paragraphs."""
    tex = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    cap = re.search(r"\\caption(?:of\{table\})?\{", tex)          # a float's caption, or a non-floating block's
    k, b = cap.start(), cap.end() - 1
    lines, i = [], tex.index(BEGIN) if tex.find("\\par{") == -1 or tex.find("\\par{") > tex.index(BEGIN) else tex.index("\\par{")
    while True:
        a = tex.find(BEGIN, i)
        if a == -1 or a > k:
            break
        between = re.sub(r"\\par\\(small|med|big)skip|\\vspace\{[^}]*\}|\\par(?![A-Za-z])", " ", tex[i:a])
        if plain(between):
            lines += ([""] if lines else []) + [plain(between), ""]
        elif lines:
            lines.append("")
        table, i = _tabular(tex, a)
        lines += table
    return lines, plain(tex[b + 1:_group(tex, b) - 1])


def write(name: str, ex=None) -> Path:
    """Write ``tables/<name>.md`` next to ``tables/<name>.tex`` and register it as an output of `ex`."""
    tex = (C.HERE / "tables" / f"{name}.tex").read_text()
    floats = [m.group(0) for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.S)] or [tex]
    blocks = []
    for chunk in floats:                             # a table that continues on a second float: one block per float
        lines, caption = convert(chunk)
        blocks.append("\n".join(lines) + "\n\n" + caption)
    path = C.HERE / "tables" / f"{name}.md"
    path.write_text(f"<!-- Generated from {name}.tex by `{C.COMMAND} {name}`; the LaTeX file is the reference. -->\n\n"
                    + "\n\n---\n\n".join(blocks) + "\n")
    if ex is not None:
        ex.output(path)
    return path
