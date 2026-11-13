"""Build the paper's tables and figures from the saved results.

    python paper-artifacts/build.py                      # everything, then the build report and one preview PDF
    python paper-artifacts/build.py list
    python paper-artifacts/build.py table_2_primary_id fig1_rank_composition

Run from the repository root. Everything is written under paper-artifacts/ (tables/, figures/, data/,
derived/, reports/). Nothing is fitted or rerun: the exhibits read the raw results under results/, which
the notebooks write, and `manifest.toml` says which saved run feeds which cell.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C                      # noqa: E402
import new_tables                       # noqa: E402
import regenerated                      # noqa: E402
import tables_md                        # noqa: E402
from paper_assets.lib import latex      # noqa: E402

#: name -> (group, kind, title, builder). Build order: the exhibits built with the first-draft code (Table B1
#: compares its cells with the figures' data files, so it follows them), then the tables built here.
REGEN_TITLES = {"tableA1_dataset_composition": ("table", "Table A1: dataset composition and linked groups"),
                "tableA2_encoders": ("table", "Table A2: encoders"),
                "table_2_primary_id": ("table", "Table 2: the in-distribution comparison, seeds 0-4"),
                "fig1_rank_composition": ("figure", "Figure 1: rank and composition, seeds 0-4"),
                "fig2_geometry_discovery": ("figure", "Figure 2: geometry and discovery, seeds 0-4"),
                "tableB1_extended_id_block": ("table", "Table B1: the extended in-distribution comparison, seeds 0-4"),
                "figB3_frozen_transfer_rank": ("figure", "Figure B3: frozen transfer, seeds 0-4, with displacement-matched PCA")}


def registry() -> dict:
    out = {n: ("regenerated", *REGEN_TITLES[n], fn) for n, fn in regenerated.EXHIBITS.items()}
    out.update({n: ("new", "table", title, fn) for n, (fn, title) in new_tables.TABLES.items()})
    return out


def _float_file(name: str, kind: str) -> Path:
    return HERE / ("tables" if kind == "table" else "figures") / (f"{name}.tex" if kind == "table" else f"{name}.include.tex")


def build_one(name: str, group: str, kind: str, title: str, fn) -> dict:
    if group == "regenerated":
        return fn()
    with C.Exhibit(name, kind, title) as ex:
        fn(ex)
        if kind == "table":
            C.preview(name, ex)
            tables_md.write(name, ex)
        else:                           # the figure with its caption on the TACL page
            latex.compile_preview(name, _float_file(name, kind).read_text(), ex)
        return C.finish(ex, name)


def preview_all(reg: dict) -> None:
    """One PDF with every exhibit of the update on TACL pages."""
    import re
    parts = []
    for name, (group, kind, title, _) in reg.items():
        f = _float_file(name, kind)
        if f.exists():
            body = f.read_text()
            if name in C.FULL_WIDTH:              # a non-floating block typeset in a one-column region
                body = "\\onecolumn\n" + body + "\n\\twocolumn\n"
            parts.append(body + "\n\\clearpage\n")
    body = "\n".join(parts)
    defined = set(re.findall(r"\\label\{([^}]+)\}", body))
    stubs = {k: v for k, v in {**C.MANUSCRIPT_LABELS, **C.UPDATE_LABELS}.items() if k not in defined}
    pre = "\\makeatletter\n" + "".join(f"\\@namedef{{r@{k}}}{{{{{v}}}{{}}}}\n" for k, v in stubs.items()) + "\\makeatother\n"
    with C.Exhibit("preview_all", "report") as ex:
        latex.compile_preview("preview_all", pre + body, ex)
        errors = [i for i in ex.issues if i["level"] in ("error", "missing")]
    print(f"[preview_all] {'ok' if not errors else f'{len(errors)} errors'} -> {C.REL}/reports/previews/preview_all.pdf")
    for i in errors:
        print("    ", i["code"], "|", i["message"][:200])


def report(reg: dict, status: dict) -> Path:
    lines = ["# Build report", "",
             f"Rebuild: `{C.COMMAND}`. One row per exhibit; a check is an assertion about the saved results that the "
             "exhibit records in its provenance file (`data/<name>.provenance.json`).", "",
             "| Exhibit | Group | Kind | Checks | Errors | Warnings | Title |", "|---|---|---|---:|---:|---:|---|"]
    for name, (group, kind, title, _) in reg.items():
        st = status.get(name)
        if st is None:
            continue
        lines.append(f"| `{name}` | {group} | {kind} | {st['checks']} | {len(st['errors'])} | {len(st['warnings'])} | {title} |")
    bad = {n: st for n, st in status.items() if st["errors"]}
    if bad:
        lines += ["", "## Errors", ""]
        for n, st in bad.items():
            lines += [f"- `{n}`: {i['code']}: {i['message']}" for i in st["errors"]]
    warn = sorted({(i["code"], i["message"]) for st in status.values() for i in st["warnings"]})
    if warn:
        lines += ["", "## Warnings", ""] + [f"- {c}: {m}" for c, m in warn]
    p = HERE / "reports" / "build_report.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    reg = registry()
    if argv == ["list"]:
        for name, (group, kind, title, _) in reg.items():
            print(f"{name:42s} {group:12s} {kind:7s} {title}")
        return 0
    unknown = [n for n in argv if n not in reg]
    if unknown:
        print(f"unknown exhibit(s): {unknown}; see `{C.COMMAND} list`")
        return 2
    names = argv or list(reg)
    status, failed = {}, 0
    with C.profile():
        regenerated.derive_band_slots()
        for name in names:
            try:
                st = build_one(name, *reg[name])
            except Exception:
                failed += 1
                print(f"[{name}] EXCEPTION")
                traceback.print_exc(limit=6)
                continue
            status[name] = st
            print(f"[{name}] {st['checks']} checks passed, {len(st['errors'])} errors/missing, "
                  f"{len(st['warnings'])} warnings")
            for i in st["errors"][:8]:
                detail = {k: v for k, v in i.items() if k not in ("level", "code", "message", "exhibit")}
                print("    ", i["level"], i["code"], "|", i["message"][:220], "|", str(detail)[:400])
        if not argv:
            preview_all(reg)
            print(f"report -> {report(reg, status).relative_to(C.REPO)}")
    n_err = sum(len(st["errors"]) for st in status.values())
    print(f"{len(status)} exhibits built, {n_err} errors, {failed} failures")
    return 1 if (n_err or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
