"""Fetch the bilingual evaluation corpus from Wikipedia (CC BY-SA 4.0).

Why a snapshot instead of live calls at query time:
- Reproducibility: the same revision is ingested on every machine.
- Offline operation: the demo never depends on Wikipedia being reachable.
- Attribution: page URL, revision id and license are stored per document.

Usage:
    python scripts/fetch_corpus.py            # fetch missing documents
    python scripts/fetch_corpus.py --force    # re-fetch everything
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
MANIFEST = CORPUS / "manifest.json"

PAGES = [
    {"id": "ar-riyadh", "lang": "ar", "title": "الرياض"},
    {"id": "ar-riyadh-metro", "lang": "ar", "title": "قطار الرياض"},
    {"id": "ar-desalination", "lang": "ar", "title": "تحلية مياه"},
    {"id": "ar-gezira", "lang": "ar", "title": "جامعة الجزيرة"},
    {"id": "ar-ai", "lang": "ar", "title": "ذكاء اصطناعي"},
    {"id": "ar-khartoum", "lang": "ar", "title": "الخرطوم"},
    {"id": "en-riyadh-metro", "lang": "en", "title": "Riyadh Metro"},
    {"id": "en-desalination", "lang": "en", "title": "Desalination"},
    {"id": "en-khartoum", "lang": "en", "title": "Khartoum"},
    {"id": "en-saudi-water", "lang": "en", "title": "Water supply and sanitation in Saudi Arabia"},
]

USER_AGENT = "SanadRAG/1.0 (https://github.com/MuaazEid/sanad)"


def fetch(lang: str, title: str) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|revisions|info",
        "explaintext": "1",
        "exlimit": "1",
        "rvprop": "ids|timestamp",
        "rvslots": "main",
        "redirects": "1",
        "inprop": "url",
        "titles": title,
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.load(resp)
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or not page.get("extract"):
        raise RuntimeError(f"no extract for {lang}:{title}")
    rev = (page.get("revisions") or [{}])[0]
    return {
        "title": page["title"],
        "text": page["extract"],
        "url": page.get("fullurl", ""),
        "revision": rev.get("revid"),
        "revision_timestamp": rev.get("timestamp"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    CORPUS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"documents": []}
    known = {d["id"]: d for d in manifest["documents"]}

    for spec in PAGES:
        target = CORPUS / f"{spec['id']}.txt"
        if target.exists() and not args.force and spec["id"] in known:
            print(f"skip  {spec['id']} (exists)")
            continue
        try:
            page = fetch(spec["lang"], spec["title"])
        except Exception as exc:  # network/API errors should not corrupt the manifest
            print(f"FAIL  {spec['id']}: {exc}")
            continue
        target.write_text(page["text"], encoding="utf-8")
        known[spec["id"]] = {
            "id": spec["id"],
            "lang": spec["lang"],
            "title": page["title"],
            "path": f"corpus/{spec['id']}.txt",
            "source_url": page["url"],
            "revision": page["revision"],
            "revision_timestamp": page["revision_timestamp"],
            "license": "CC BY-SA 4.0",
        }
        print(f"ok    {spec['id']}: {len(page['text'])} chars, rev {page['revision']}")
        time.sleep(0.6)  # be polite to the API

    manifest["documents"] = [known[s["id"]] for s in PAGES if s["id"] in known]
    manifest["corpus_license"] = "CC BY-SA 4.0 (text); code is MIT"
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {len(manifest['documents'])} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
