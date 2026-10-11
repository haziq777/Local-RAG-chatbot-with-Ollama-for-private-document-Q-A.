"""Parse the price-list PDF into rows and build a local embedding index."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PDF = HERE / "Price-List-2026_Digital-FINAL-17.08.2026.pdf"
DATA = HERE / "data"
MODEL = "all-MiniLM-L6-v2"

ROW = re.compile(
    r"^(?P<desc>.+?)\s{2,}(?P<disc>[A-Z]\d{1,2})\s+(?P<code>\S+)"
    r"(?:\s+(?P<pack>\d+))?(?:\s+(?P<ctn>\d+))?\s+(?P<price>\d{1,3}(?:,\d{3})*\.\d{2})\s*$"
)
HEADER = re.compile(r"^\s*DESCRIPTION\b")
SKIP = re.compile(r"^\s*(INDEX\b|PACK\s*/|DISC\b|CODE\s|INNER|QTY|HR PRODUCTS PRICE LIST)")


def page_count():
    out = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def page_text(n):
    return subprocess.run(
        ["pdftotext", "-f", str(n), "-l", str(n), "-layout", str(PDF), "-"],
        capture_output=True, text=True, check=True,
    ).stdout


FOOTER = re.compile(r"List Prices do not include.*?GST\s+(\S+)\s*$")


def footer_label(lines):
    """Printed page number from the page footer (e.g. '134' or 'IX'); None if the page has none."""
    for l in reversed(lines[-6:]):
        m = FOOTER.search(l)
        if m:
            return m.group(1)
    return None


def column_offsets(lines):
    """Character offsets of the PACK/INNER and CTN/OUTER columns on a page."""
    for l in lines:
        if "PACK" in l and "CTN" in l:
            return l.index("PACK"), l.index("CTN")
    pairs = [(m.start("pack"), m.start("ctn")) for m in map(ROW.match, (l.rstrip() for l in lines))
             if m and m["pack"] and m["ctn"]]
    if pairs:
        return sorted(p for p, _ in pairs)[len(pairs) // 2], sorted(c for _, c in pairs)[len(pairs) // 2]
    return None


def split_qty(m, cols):
    """The row regex puts a lone quantity in `pack`; use its column to decide which one it is."""
    pack, ctn = m["pack"], m["ctn"]
    if pack and not ctn and cols and m.start("pack") >= cols[1] - 3:
        return None, pack
    return pack, ctn


def parse():
    rows, skipped = [], 0
    for n in range(1, page_count() + 1):
        section, sub, seen_header = "", "", False
        lines = [l for l in page_text(n).splitlines() if l.strip()]
        cols = column_offsets(lines)
        label = footer_label(lines)
        prev_text = ""  # last unmatched mixed-case line, used when a row has no inline description
        for idx, line in enumerate(lines):
            if HEADER.match(line):
                seen_header = True
                continue
            m = ROW.match(line.rstrip())
            if m:
                desc = re.sub(r"\s+", " ", m["desc"]).strip()
                repaired = False
                if not desc and prev_text:
                    desc, repaired = prev_text, True
                    nxt = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
                    if nxt.lower().startswith("c/w") and not ROW.match(lines[idx + 1].rstrip()):
                        desc = f"{desc} {nxt}"
                pack, ctn = split_qty(m, cols)
                rows.append({
                    "description": desc,
                    "disc_code": m["disc"],
                    "supplier_code": m["code"],
                    "pack_qty": pack,
                    "ctn_qty": ctn,
                    "price": float(m["price"].replace(",", "")),
                    "section": section,
                    "subheading": sub,
                    "page": label or str(n),  # printed page number, as in the PDF index
                    "pdf_page": n,  # position in the PDF viewer
                    "repaired": repaired,
                    "context": [l.strip() for l in lines[max(0, idx - 3): idx + 4]] if repaired else None,
                })
                prev_text = ""
            elif SKIP.match(line) or form_noise(line):
                continue
            elif not seen_header and not section:
                section = line.strip()
            elif seen_header and line.strip().isupper() and len(line.strip()) < 100:
                sub = line.strip()
                prev_text = ""
            else:
                prev_text = re.sub(r"\s+", " ", line).strip()
                skipped += 1
    return rows, skipped


def form_noise(line):
    return len(line.strip()) < 3


def llm_repair(rows):
    """Let the local LLM tidy descriptions that were rebuilt from wrapped lines (cached; skipped if no Ollama)."""
    import llm

    todo = [r for r in rows if r["repaired"]]
    if not todo or not llm.available():
        print("LLM repair skipped (Ollama/model not available)")
        return
    cache_file = DATA / "repairs.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    changed = 0
    for r in todo:
        key = f'{r["pdf_page"]}:{r["supplier_code"]}'
        if key not in cache:
            cache[key] = llm.repair_description(r["context"], r["supplier_code"])
        if cache[key] and cache[key] != r["description"]:
            r["description"] = cache[key]
            changed += 1
    DATA.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=1))
    print(f"LLM repair: {changed} of {len(todo)} descriptions updated")


def main():
    DATA.mkdir(exist_ok=True)
    rows, skipped = parse()
    print(f"parsed {len(rows)} rows; {skipped} unmatched lines; "
          f"{sum(r['repaired'] for r in rows)} descriptions rebuilt from neighbouring lines")
    llm_repair(rows)
    for r in rows:
        r.pop("context", None)
    (DATA / "items.json").write_text(json.dumps(rows, indent=1))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    texts = [f'{r["section"]} | {r["subheading"]} | {r["description"]} | {r["supplier_code"]}' for r in rows]
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.save(DATA / "embeddings.npy", emb.astype("float32"))
    print("saved index")


if __name__ == "__main__":
    main()
