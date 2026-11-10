"""Derived KB release = an existing kb/chunks/v1 release + the NSU handbook prepared for Menon (nsu_main_info.md,
19.10.2026, from nsu.ru materials), one document per «###» subsection (the «##» intro is its own document), chunked by
the release's own tools/chunk_kb.py and indexed by the same BM25 normalisation (tools/rag_bm25.py). Nothing of the
handbook is dropped. Document URL: the subsection's own «Источник: [..](url)» link, else the section's link in the
handbook's table of contents; label «справочник НГУ для Менона (по материалам nsu.ru, 19.10.2026)».
usage: KB_FRIDA_TOKENIZER=<frida> python build_handbook.py OLD_ROOT HANDBOOK_MD NEW_ROOT SOURCE_URL"""
import gzip, hashlib, json, os, re, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

OLD, HB, NEW = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SRC_URL = sys.argv[4]
OV, NV = OLD / "kb/chunks/v1", NEW / "kb/chunks/v1"
BUILD, SITE, LABEL, TIER, DTYPE, DATE = ("nsu_handbook_2026_10_19", "handbook.nsu", "справочник НГУ для Менона (по материалам nsu.ru, 19.10.2026)",
                                         "P1", "handbook", "2026-10-19")
LINK = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for f in src.iterdir():
            link(f, dst / f.name)
    else:
        os.link(src, dst)


def sections(md):
    """[(section_no, section_title, sub_title or None, text)]; ToC links per section number"""
    toc = {m.group(1): m.group(2) for m in re.finditer(r"(?m)^(\d+)\. \[[^\]]+\]\((https?://[^)]+)\)", md)}
    out = []
    for sec in re.split(r"(?m)^(?=## )", md):
        m = re.match(r"## (?:(\d+)\.\s*)?(.+)", sec)
        if not m:
            continue
        no, stitle = m.group(1), m.group(2).strip()
        if stitle.lower().startswith("содержание"):
            continue
        parts = re.split(r"(?m)^(?=### )", sec)
        intro = parts[0].strip()
        if len(re.sub(r"(?m)^## .*$", "", intro).strip()) > 40:
            out.append((no, stitle, None, intro))
        for p in parts[1:]:
            sub = re.match(r"### (.+)", p).group(1).strip()
            out.append((no, stitle, sub, f"## {stitle}\n\n" + p.strip()))
    return out, toc


def main():
    t0 = time.time()
    if NEW.exists():
        sys.exit(f"{NEW} exists")
    md = HB.read_text(encoding="utf-8")
    secs, toc = sections(md)
    src_root = NEW / "sources_nsu_handbook"
    src_root.mkdir(parents=True)
    manifest = {"source": SRC_URL, "file": HB.name, "sha256": hashlib.sha256(md.encode()).hexdigest(), "modified": "2026-10-19T13:51:26+00:00",
                "split": "one document per ### subsection; the ## intro is its own document; text verbatim",
                "documents": len(secs), "label": LABEL, "tier": TIER}
    (src_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    os.environ["KB_SOURCE_ROOT"] = str(src_root)
    sys.path.insert(0, str(OV / "tools"))
    import chunk_kb
    import rag_bm25
    base_label = chunk_kb.org_label
    chunk_kb.org_label = lambda site: LABEL if site == SITE else base_label(site)
    chunk_kb.init()
    for item in OV.iterdir():
        if item.name not in ("docs.jsonl.gz", "stats.json", "README.md", "ragcore"):
            link(item, NV / item.name)
    old_stats = json.loads((OV / "stats.json").read_text())
    parts = sorted((OV / "chunks").glob("part-*.jsonl.gz"))
    next_part = int(parts[-1].name[5:10]) + 1
    mapping = json.load(gzip.open(OV / "ragcore/chunk_mapping_to_texts.json.gz", "rt"))
    gi, di = old_stats["indexed"], old_stats["docs"]
    assert len(mapping) == gi
    src_sha = chunk_kb.SOURCE_MANIFEST_SHA256
    seen, new_chunks, new_docs, new_corpus = {}, [], [], []
    n_low = n_dup = 0
    for k, (no, stitle, sub, text) in enumerate(secs):
        title = f"Справочник НГУ: {stitle}" + (f" — {sub}" if sub else "")
        own = re.search(r"Источник: \[[^\]]*\]\((https?://[^)\s]+)\)", text)
        url = own.group(1) if own else (toc.get(no) or f"kb://nsu-handbook/{k}")
        doc_id = "kb-" + hashlib.sha256(f"{BUILD}:{k}:{title}".encode()).hexdigest()[:32]
        ts = hashlib.sha256(text.encode()).hexdigest()
        ver = hashlib.sha256((doc_id + "\0" + ts).encode()).hexdigest()[:32]
        m = {"id": doc_id, "kb_version_id": ver, "text_sha256": ts, "title": title, "site": SITE, "kb_build": BUILD,
             "canonical_url": url, "priority": {"tier": TIER}, "dates": {"published": DATE}, "semantic": {"doc_type": DTYPE}}
        r = chunk_kb.process((m, text))
        if r is None:
            continue
        stem = f"{doc_id}#v{ver}#c"
        rc, ids = [], []
        for ci, ch in enumerate(r["chunks"]):
            if ch["flag"] == "over_limit":
                raise ValueError("FRIDA token ceiling exceeded: " + stem)
            cid = f"{stem}{ci:03d}"; ids.append(cid)
            key = hashlib.sha256(re.sub(r"\s+", " ", ch["text"].lower()).strip().encode()).hexdigest()
            dup = seen.get(key)
            indexed = not dup and not ch["low_content"]
            n_dup += bool(dup); n_low += ch["low_content"]
            if indexed:
                seen[key] = cid
                mapping[str(gi)] = {"doc_index": di, "local_chunk_index": len(rc)}; gi += 1
                rc.append({"start_char": ch["start_char"], "end_char": ch["end_char"], "num_tokens": ch["tokens"], "chunk_id": cid})
            new_chunks.append({
                "chunk_id": cid, "doc_id": doc_id, "kb_build": BUILD, "chunk_index": ci, "n_chunks": len(r["chunks"]),
                "source_version_id": ver, "source_text_sha256": ts, "source_manifest_sha256": src_sha,
                "start_char": ch["start_char"], "end_char": ch["end_char"], "text": ch["text"], "index_text": ch["index_text"],
                "source_spans": ch["source_spans"], "source_joiner": ch["source_joiner"], "heading_path": ch["heading_path"],
                "tokens_frida": ch["tokens"], "tokens_index_frida": ch["tokens_index"], "has_table": ch["has_table"],
                "table_header_repeated": ch["table_header_repeated"], "indexed": indexed,
                "not_indexed_reason": None if indexed else ("duplicate_chunk" if dup else "low_content"), "dup_of": dup,
                "low_content": ch["low_content"], "title": r["title"], "url": url, "site": SITE, "source_label": LABEL,
                "tier": TIER, "doc_type": DTYPE, "topics": [], "date": DATE, "language": "ru", "quality_score": 0.9,
                "prev_chunk_id": f"{stem}{ci - 1:03d}" if ci else None,
                "next_chunk_id": f"{stem}{ci + 1:03d}" if ci + 1 < len(r["chunks"]) else None})
        binding = {"source_version_id": ver, "source_text_sha256": ts, "source_manifest_sha256": src_sha}
        ann = f"Раздел справочника НГУ для Менона (19.10.2026, по материалам nsu.ru): {stitle}" + (f" — {sub}" if sub else "") + "."
        new_docs.append({"doc_id": doc_id, "kb_build": BUILD, "title": r["title"], "annotation": ann, "urls": [url], "tier": TIER,
                         **binding, "doc_type": DTYPE, "quality_score": 0.9, "n_chunks": len(r["chunks"]), "n_indexed": len(rc),
                         "doc_tokens_frida": r["doc_tokens"], "text_rendered": r["rendered"], "chunk_ids": ids})
        new_corpus.append({"content_source": BUILD, "kb_id": doc_id, **binding, "url": url, "doc_title": r["title"],
                           "doc_annotation": ann, "doc_full_text": r["full"], "chunks": rc, "num_tokens": r["doc_tokens"],
                           "quality_score": 0.9, "tier": TIER, "doc_type": DTYPE, "date": DATE})
        di += 1

    def write_gz(path, rows, base=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if base is not None:
            shutil.copyfile(base, path)
        with gzip.open(path, "at", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_gz(NV / f"chunks/part-{next_part:05d}.jsonl.gz", new_chunks)
    write_gz(NV / "docs.jsonl.gz", new_docs, OV / "docs.jsonl.gz")
    write_gz(NV / "ragcore/chunked_texts_about_nsu_with_metadata.jsonl.gz", new_corpus, OV / "ragcore/chunked_texts_about_nsu_with_metadata.jsonl.gz")
    with gzip.open(NV / "ragcore/chunk_mapping_to_texts.json.gz", "wt") as f:
        json.dump(mapping, f)
    old_rev = (OLD / "REVISION").read_text().strip()
    (NEW / "REVISION").write_text(f"{old_rev}+nsu-handbook-2026-10-19\n")
    stats = {**old_stats, "chunks": old_stats["chunks"] + len(new_chunks), "docs": di, "indexed": gi,
             "unique_document_ids": old_stats["unique_document_ids"] + len(new_docs),
             "unique_chunk_ids": old_stats["unique_chunk_ids"] + len(new_chunks),
             "nsu_handbook": {"documents": len(new_docs), "chunks": len(new_chunks), "indexed": sum(c["indexed"] for c in new_chunks),
                              "duplicate": n_dup, "low_content": n_low}}
    (NV / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    (NV / "README.md").write_text(f"# Производный выпуск: {old_rev} + справочник НГУ для Менона (19.10.2026)\n\nИсточник и разбиение: "
                                  "`sources_nsu_handbook/manifest.json`.\n\n" + (OV / "README.md").read_text(encoding="utf-8"))
    print(json.dumps(stats["nsu_handbook"], ensure_ascii=False), round(time.time() - t0), "s", flush=True)
    import bm25s
    ordering, normalized = hashlib.sha256(), []

    def texts():
        for ch in rag_bm25.chunks(NV):
            ordering.update((ch["chunk_id"] + "\n").encode())
            yield ch["index_text"]

    with Pool(16, initializer=rag_bm25.init) as pool:
        for res in pool.imap(rag_bm25.normalize, texts(), chunksize=256):
            normalized.append(res)
    tokenized = bm25s.tokenize(normalized, stemmer=None, stopwords=[], show_progress=False)
    n = len(normalized); del normalized
    index = bm25s.BM25(); index.index(tokenized, show_progress=False)
    dest = NV / "ragcore/knowledge/bm25"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    index.save(str(dest), show_progress=False)
    (dest / "meno_kb_build.json").write_text(json.dumps({"n_chunks": n, "indexed_chunk_order_sha256": ordering.hexdigest(),
                                                          "bm25s_version": bm25s.__version__}, indent=2) + "\n")
    assert n == gi, (n, gi)
    print("bm25", n, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
