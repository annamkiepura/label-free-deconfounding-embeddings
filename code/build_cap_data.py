"""Rebuild the parent paper's CAP corpora: bills, executive orders, newspapers.

The category-level experiment of Fan et al. (2025, Sec. 3.1) clusters documents
from three Comparative Agendas Project (CAP) sources against the 21 CAP major
topics. The parent repository ships none of the raw files -- its
`erasure_CAP.ipynb` reads them from its working directory -- so this script
rebuilds each one from its public source:

* **Newspapers** -- CAP's Pennsylvania Policy Database "Newspaper Articles"
  (67,963 clips, 1979-2015; the paper's footnote 4). The text is the headline
  and the abstract joined by a newline, as in the parent notebook.
* **Bills** -- the labelled congressional bill summaries of Hoyle et al.
  (2022), train and test concatenated, as in the parent notebook. Replaying the
  notebook's per-topic sampling on these two sources reproduces the batch
  counts in its committed outputs exactly, so they are the files it used.
* **Executive orders** -- the topic codes of CAP US Executive Orders v21.3, the
  version the parent used, joined on the order number to each order's full
  text from the Federal Register API. The parent's own full-text file was never
  released. The Federal Register has full text from EO 12890 (December 1993)
  onward, so earlier orders are not covered; the parent's committed outputs
  imply its file did not cover them either. v21.3 has been retired upstream
  (the live release is 26.1), so it is fetched from the Wayback Machine.

An order's text keeps what the parent's extraction regex kept -- "Executive
Order N of <date>" through "THE WHITE HOUSE, <date>." -- minus the Federal
Register furniture (page markers, signature and graphic placeholders). It stops
at the signature date, where the parent's ran on to the end of the document and
so also took any annex. CAP never codes an executive order 23 (Culture), so
this source spans at most 20 topics.

Every row carries one of the 21 major topics; any other code (0, 24, 99,
missing) is dropped, as in the parent notebook. Nothing is deduplicated or
sampled here: the loaders drop exact duplicates at load time and the protocol
does the sampling.

Writes ``data/cap_{bills,orders,newspapers}.jsonl.gz`` with the columns id,
source, year, majortopic, topic, text; ``pd.read_json(path, lines=True)`` reads
them as they are. They are gzipped because the bill summaries alone are 81 MB
of text, and with a fixed timestamp, so a rebuild from the same sources is
byte-identical. Raw downloads are cached under ``data/raw/cap/`` (in the repository except the two
bill files), so a rebuild only fetches what is missing, and the bulk files are checked
against pinned SHA-256 digests, so an upstream change fails loudly instead of
silently changing the corpus.

Also writes ``data/cap_sample.jsonl``, the three sources subsampled to the
sizes of the parent paper's Table 1 (see `table1_sample`): each pairing the
paper evaluates is this file filtered to its sources.

Usage, from the repository root::

    python code/build_cap_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

#: The 21 CAP major topics. The names are the parent notebook's, and they are
#: exactly the ``topic`` strings of Hoyle et al.'s bills.
MAJOR_TOPICS = {
    1: "Macroeconomics", 2: "Civil Rights", 3: "Health", 4: "Agriculture",
    5: "Labor", 6: "Education", 7: "Environment", 8: "Energy",
    9: "Immigration", 10: "Transportation", 12: "Law and Crime",
    13: "Social Welfare", 14: "Housing", 15: "Domestic Commerce",
    16: "Defense", 17: "Technology", 18: "Foreign Trade",
    19: "International Affairs", 20: "Government Operations",
    21: "Public Lands", 23: "Culture",
}

#: Bulk sources: cache file name -> (url, sha256 of the file the corpus was built from).
SOURCES = {
    "CAP_NewspaperClips1979_2015.csv": (
        "https://minio.la.utexas.edu/compagendas/datasetfiles/CAP_NewspaperClips1979_2015.csv",
        "ccdcba5fa0314151deb4929370eb72c49546a1d414de3a9b82e76bbdd6771536",
    ),
    "bills_train.metadata.jsonl": (
        "https://umd-clip-public.s3.amazonaws.com/topics-emnlp.findings-2022/bills-labeled/train.metadata.jsonl",
        "d0e21141a854b81b25cd1898022fd62b3277bf5142278b78ce58fbf05efbc492",
    ),
    "bills_test.metadata.jsonl": (
        "https://umd-clip-public.s3.amazonaws.com/topics-emnlp.findings-2022/bills-labeled/test.metadata.jsonl",
        "ed978983dc8c5306b1d3253816dd498a8815efede2afcc8b4cc21d8fd1174a46",
    ),
    "US-Executive-executive_orders_21.3.csv": (
        "https://web.archive.org/web/20241213093344id_/"
        "https://minio.la.utexas.edu/compagendas/datasetfiles/US-Executive-executive_orders_21.3.csv",
        "81e0248e2ca2377cf95fc9405b91cfd57e92dfc97ac08327426d2cefc499f0c9",
    ),
}

FR_API = "https://www.federalregister.gov/api/v1/documents.json"
FR_FIELDS = ("executive_order_number", "document_number", "publication_date", "raw_text_url")
USER_AGENT = "clean-embeddings CAP corpus builder"
COLUMNS = ["id", "source", "year", "majortopic", "topic", "text"]

#: The parent paper's Table 1: documents per CAP source pairing (all 21 topics).
PARENT_TABLE1 = {
    ("bill", "order"): 1902,
    ("bill", "newspaper"): 2613,
    ("order", "newspaper"): 1907,
    ("bill", "order", "newspaper"): 3211,
}
#: Documents per topic and source in the parent's sample (see `table1_sample`).
PER_TOPIC = 64


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def _get(url, tries=5):
    """GET `url`, backing off on rate limits, server errors and dropped connections."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
        time.sleep(5 * 2 ** attempt)


def _write(path, data):
    """Write via a temporary file, so an interrupted download never looks cached."""
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)


def _cached(name, cache):
    """Path to the bulk source `name`, downloaded once and checked against its pin."""
    url, sha256 = SOURCES[name]
    path = cache / name
    if not path.exists():
        print(f"downloading {url}")
        _write(path, _get(url))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(
            f"{path} has SHA-256 {digest}, expected {sha256}. The upstream file has "
            "changed; inspect it and update the pin in SOURCES deliberately."
        )
    return path


# --------------------------------------------------------------------------- #
# The three sources -> id, source, year, majortopic, topic, text
# --------------------------------------------------------------------------- #
def build_newspapers(cache):
    df = pd.read_csv(_cached("CAP_NewspaperClips1979_2015.csv", cache))
    df = df.dropna(subset=["year", "headline", "abstract"])
    df = df[df["majortopic"].isin(list(MAJOR_TOPICS))]
    code = df["majortopic"].astype(int)
    return pd.DataFrame({
        "id": df["id"].astype(str),
        "source": "newspaper",
        "year": df["year"].astype(int),
        "majortopic": code,
        "topic": code.map(MAJOR_TOPICS),
        "text": df["headline"].astype(str) + "\n" + df["abstract"].astype(str),
    })


def build_bills(cache):
    df = pd.concat(
        [pd.read_json(_cached(f"bills_{split}.metadata.jsonl", cache), lines=True)
         for split in ("train", "test")],
        ignore_index=True,
    )
    codes = {name: code for code, name in MAJOR_TOPICS.items()}
    df = df[df["topic"].isin(list(codes)) & df["summary"].fillna("").str.strip().ne("")]
    return pd.DataFrame({
        "id": df["id"].astype(str),
        "source": "bill",
        "year": pd.to_datetime(df["date"]).dt.year,
        "majortopic": df["topic"].map(codes),
        "topic": df["topic"],
        "text": df["summary"],
    })


def _fr_listing(cache):
    """Every executive order the Federal Register API lists, cached on first use."""
    path = cache / "federal_register_executive_orders.json"
    if not path.exists():
        query = urllib.parse.urlencode(
            [("conditions[presidential_document_type]", "executive_order"),
             ("per_page", 1000), ("order", "oldest")]
            + [("fields[]", field) for field in FR_FIELDS]
        )
        url, rows = f"{FR_API}?{query}", []
        while url:
            page = json.loads(_get(url))
            rows += page["results"]
            url = page.get("next_page_url")
        _write(path, json.dumps(rows).encode())
    return pd.DataFrame(json.loads(path.read_text()))


def _fr_text(doc, cache):
    """Raw Federal Register text of one document, cached by document number."""
    path = cache / "federal_register" / f"{doc['document_number']}.txt"
    if not path.exists():
        _write(path, _get(doc["raw_text_url"]))
    return path.read_bytes().decode("utf-8", errors="replace")


#: GPO spells accented letters as bracketed codes (``C[ocirc]te``); only these
#: are rewritten, since EO text also has real editorial brackets (``[or]``).
_DIACRITICS = {
    "acute": "\u0301", "grave": "\u0300", "circ": "\u0302", "tilde": "\u0303",
    "uml": "\u0308", "ring": "\u030a", "cedil": "\u0327", "breve": "\u0306",
    "macr": "\u0304", "caron": "\u030c",
}
_DATE = r"[a-z]+\.?\s+\d{1,2},?\s*\d{4}"


def _order_text(raw, number):
    """The body of executive order `number` from its Federal Register text, or None.

    The file is HTML around a <pre> block of the printed page. GPO leaves its
    placeholders (``<GRAPHIC(S) NOT AVAILABLE IN TIFF FORMAT>``, signature codes
    such as ``<Clinton1>><Clinton2>``) as bare tags and escapes real angle
    brackets, so the bare tags are dropped before entities are unescaped.

    The order starts at its header, "Executive Order [No.] N of <date>", where
    the number must be followed by "of" or a date: a table of contents lists the
    order as "Executive Order N--<title>", and a few headers drop the "of" or
    misprint the date. It ends at "THE WHITE HOUSE, [Washington,] <date>.". The
    printed lines are hard-wrapped, so each paragraph is re-joined into one line.
    """
    pre = re.search(r"(?is)<pre>(.*)</pre>", raw)
    body = html.unescape(re.sub(r"<[^<>\n]*>+", " ", pre.group(1) if pre else raw))
    start = re.search(rf"(?i)executive\s+order\s+(?:no\.\s*)?{number}\s+(?:of\b|{_DATE})", body)
    if start is None:
        return None
    body = body[start.start():]
    end = re.search(rf"(?i)the\s+white\s+house,\s*(?:washington,\s*)?{_DATE}\.?", body)
    body = body[:end.end()] if end else re.split(r"\[FR Doc|Billing code", body, maxsplit=1)[0]
    # The signature placeholder often carries a code glued to it: "(Presidential Sig.)B".
    body = re.sub(r"\[\[Page [^\]]*\]\]|\[GRAPHIC[^\]]*\]|\(Presidential Sig\.\)\S*", " ", body)
    body = re.sub(rf"\[([A-Za-z])({'|'.join(_DIACRITICS)})\]",
                  lambda m: m.group(1) + _DIACRITICS[m.group(2)], body)
    body = body.replace("``", '"').replace("''", '"')
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", body))
    return unicodedata.normalize("NFC", "\n".join(p for p in paragraphs if p))


def build_orders(cache, workers=4):
    cap = pd.read_csv(_cached("US-Executive-executive_orders_21.3.csv", cache))
    # v21.3 lists EO 12627 twice with the same code; keep one row per order.
    cap = cap.dropna(subset=["eo_number"]).drop_duplicates("eo_number")
    cap = cap[cap["majortopic"].isin(list(MAJOR_TOPICS))]

    fr = _fr_listing(cache)
    fr["eo_number"] = pd.to_numeric(fr["executive_order_number"], errors="coerce")
    # Three orders were printed twice (an R1-/C1- republication or correction
    # notice after the original); keep the original printing.
    fr = fr.dropna(subset=["eo_number"]).sort_values("publication_date")
    fr = fr.drop_duplicates("eo_number")
    df = cap.merge(fr[["eo_number", "document_number", "raw_text_url"]], on="eo_number")

    print(f"fetching {len(df):,} executive orders from the Federal Register")
    (cache / "federal_register").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        raw = list(pool.map(lambda doc: _fr_text(doc, cache), df.to_dict("records")))
    number = df["eo_number"].astype(int)
    df["text"] = [_order_text(r, n) for r, n in zip(raw, number)]
    # Three printings have no header to anchor on: EO 12910's omits the number,
    # EO 13523's misprints it as 15323, and the document listed for EO 12936 is
    # only the table of contents of the issue that carries it.
    missing = df["text"].isna()
    if missing.any():
        print(f"orders: no order body found for {int(missing.sum())} of {len(df)}: "
              + ", ".join(f"EO {n}" for n in number[missing]))
    df, number = df[~missing], number[~missing]
    code = df["majortopic"].astype(int)
    return pd.DataFrame({
        "id": number.astype(str),
        "source": "order",
        "year": df["year"].astype(int),
        "majortopic": code,
        "topic": code.map(MAJOR_TOPICS),
        "text": df["text"],
    })


def table1_sample(corpora, seed=42):
    """The three sources subsampled to the sizes of the parent paper's Table 1.

    Table 1's pair counts are additive -- (1,902 + 2,613 + 1,907) / 2 = 3,211 --
    so every pairing reuses one fixed sample per source: 1,304 bills, 598 orders
    and 1,309 newspaper clips. Taking up to 64 documents per topic reproduces
    the bills and the clips exactly (20 x 64 + all 24 Culture bills; 20 x 64 +
    all 29 Immigration clips), and no other cap does. The parent drew its orders
    the same way from a pool smaller than the Federal Register rebuild, where 64
    per topic gives 781. Which orders it lacked is unknown, so they are drawn as
    if from a random subset of the pool: shuffled, capped at 64 per topic, and
    cut at 598. The largest topics still fill to 64, as they must have for the
    parent; the shortfall falls on the smaller ones.

    Exact duplicates are dropped first, as the loaders would drop them. Only 21
    of the 24 Culture bills are distinct, so the sample has 1,301 bills, three
    short of Table 1. The sizes match the parent's; the documents are not its.
    """
    n_orders = PARENT_TABLE1[("bill", "order", "newspaper")] - PARENT_TABLE1[("bill", "newspaper")]
    parts = []
    for df in corpora.values():
        shuffled = df.drop_duplicates("text").sample(frac=1, random_state=seed)
        capped = shuffled[shuffled.groupby("topic").cumcount() < PER_TOPIC]
        parts.append(capped.head(n_orders) if df["source"].iloc[0] == "order" else capped)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=REPO / "data")
    parser.add_argument("--cache-dir", type=Path, default=REPO / "data" / "raw" / "cap")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel Federal Register downloads")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed of the Table 1 sample (the parent notebook's)")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    corpora = {
        "bills": build_bills(args.cache_dir),
        "orders": build_orders(args.cache_dir, args.workers),
        "newspapers": build_newspapers(args.cache_dir),
    }
    for name, df in corpora.items():
        path = args.out_dir / f"cap_{name}.jsonl.gz"
        df[COLUMNS].to_json(path, orient="records", lines=True, force_ascii=False,
                            compression={"method": "gzip", "mtime": 0})
        counts = df["topic"].value_counts()
        print(f"{path}: {len(df):,} documents, {len(counts)} topics "
              f"({counts.min():,}-{counts.max():,} per topic)")

    sample = table1_sample(corpora, args.seed)
    path = args.out_dir / "cap_sample.jsonl"
    sample[COLUMNS].to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"{path}: {len(sample):,} documents; against the parent's Table 1:")
    for sources, n_parent in PARENT_TABLE1.items():
        part = sample[sample["source"].isin(sources)]
        print(f"  {' + '.join(sources):26s} N = {len(part):,} (paper {n_parent:,}), "
              f"{part['topic'].nunique()} topics (paper 21)")


if __name__ == "__main__":
    main()
