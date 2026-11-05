"""Structure-aware chunking of the KB sources layer for classic RAG (Meno-Stand-RAG-Core compatible).
chunk_kb.py <out_dir> [--limit N]

Per document (RAG-ready = not duplicate, not recommended exclude; schedule snapshots of table.nsu.ru skipped):
- Markdown is parsed into blocks: headings (#..######, plus bold / ALL-CAPS single lines as weak headings), paragraphs,
  lists (items), tables (rows; header = first row + separator).
- Section tree by heading level; a section that fits MAXT FRIDA tokens is one chunk; larger sections are split into their
  intro + subsections, recursively; oversized blocks are split by sentences (razdel), list items or table rows; a single
  oversized sentence/row is cut at whitespace by token offsets. Headings never end a chunk. Small neighbours (< MINT) are merged.
- Continuation chunks of a split table get the table header repeated (the document is then "rendered": header rows inserted
  before the continuation; documents without split tables keep their text byte-identical).
- index_text = context header (title, source/organisation, date, section path) + chunk text, guaranteed to fit FRIDA's
  512 tokens together with the "search_document: " prefix.
- Identical chunk texts across documents are indexed once (dup_of), chunks with < 25 letters are dropped.
Outputs: chunks.jsonl (rich records), docs.jsonl, ragcore/chunked_texts_about_nsu_with_metadata.jsonl, ragcore/chunk_mapping_to_texts.json, stats.json"""

import collections, hashlib, json, os, re, sys, time
from pathlib import Path
from kb_release_repair import open_gzip, paired_rows
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_orgs import org_label

S = os.environ["KB_SOURCE_ROOT"]
OUT = sys.argv[1] if len(sys.argv) > 1 else None
TOKENIZER_PATH = os.environ["KB_FRIDA_TOKENIZER"]
SOURCE_MANIFEST_SHA256 = hashlib.sha256((Path(S) / "manifest.json").read_bytes()).hexdigest()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EMB_MAX, MAXT, TARGET, MINT, HDR_MAX = 512, 380, 280, 64, 72
PREFIX = "search_document: "
HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST = re.compile(r"^\s{0,3}(?:[-*+•▪●–]|\d{1,3}[.)]|[a-zа-я][.)])\s+\S")
TABLE = re.compile(r"^\s*\|")
TSEP = re.compile(r"^\s*\|?\s*:?-{2,}")
BOLDLINE = re.compile(r"^\*\*[^*\n]{2,150}\*\*:?$")
SKIP_SITES = {"table.nsu.ru"}
SKIP_TITLE = re.compile(
    r"Атлас новых исследований на основе EconLit|^Перечень рецензируемых научных изданий", re.I
)  # external bibliographic catalogues
DOC_CAP = 300  # max indexed chunks per document (P0 exempt)
SKIP_BUILDS = set(filter(None, os.environ.get("SKIP_BUILDS", "").split(",")))
TIER_Q = {"P0": 1.0, "P1": 0.9, "P2": 0.75, "P3": 0.6}
NEWSY = {"event_or_news", "news", "social_post"}
CARD_ANN = {
    "nsu_programs": "Карточка направления подготовки НГУ, составленная по официальным документам университета: профили, форма и срок обучения, места, проходные баллы, стоимость.",
    "nsu_faculties": "Карточка факультета (института) НГУ: руководство, кафедры, направления подготовки, контакты.",
    "akademgorodok_orgs": "Карточка организации Новосибирского Академгородка: история, направления работы, руководство, связь с НГУ.",
    "nsu_units": "Карточка подразделения НГУ по данным исследовательского портала Pure: место в структуре, руководство, сотрудники, исследования, публикации.",
    "nsu_people": "Карточка сотрудника НГУ по данным исследовательского портала Pure: должности, научные интересы, публикации.",
    "nsu_staff": "Карточка преподавателя НГУ: должность, кафедра, образование, преподаваемые дисциплины, научные интересы, публикации.",
}
tok = None


def init():
    global tok, sentenize
    from transformers import AutoTokenizer
    from razdel import sentenize as _s

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    tok.model_max_length = 1000000000
    sentenize = _s


def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0


def norm_title(s):
    return re.sub(r"[\W_]+", " ", (s or "").lower().replace("ё", "е")).strip()


def pseudo_heading(s):
    s = s.strip()
    if len(s) < 3 or len(s) > 120 or "\n" in s:
        return False
    if BOLDLINE.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters) and not s.endswith((".", ";", ","))


def clean_heading(s):
    return re.sub(r"\s+", " ", re.sub(r"[*_`#]+", "", s)).strip(" :")


def parse(text):
    lines, pos = [], 0
    for ln in text.split("\n"):
        lines.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1
    out, i, n = [], 0, len(lines)
    while i < n:
        s, e, ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        m = HEAD.match(ln)
        if m:
            out.append({"t": "h", "s": s, "e": e, "lvl": len(m.group(1)), "title": clean_heading(m.group(2))})
            i += 1
            continue
        if TABLE.match(ln):
            j = i
            rows = []
            while j < n and TABLE.match(lines[j][2]):
                rows.append((lines[j][0], lines[j][1]))
                j += 1
            if len(rows) >= 2 and TSEP.match(lines[i + 1][2]):
                hdr, body = (rows[0][0], rows[1][1]), rows[2:]
            else:
                hdr, body = None, rows
            out.append({"t": "table", "s": rows[0][0], "e": rows[-1][1], "hdr": hdr, "rows": body})
            i = j
            continue
        if LIST.match(ln):
            j = i
            items = []
            while j < n:
                l2 = lines[j][2]
                if LIST.match(l2):
                    items.append([lines[j][0], lines[j][1]])
                elif l2.strip() and l2[:1] in (" ", "\t") and items:
                    items[-1][1] = lines[j][1]
                elif not l2.strip() and j + 1 < n and LIST.match(lines[j + 1][2]):
                    pass
                else:
                    break
                j += 1
            out.append({"t": "list", "s": items[0][0], "e": items[-1][1], "items": [tuple(x) for x in items]})
            i = j
            continue
        j = i
        while (
            j < n
            and lines[j][2].strip()
            and not HEAD.match(lines[j][2])
            and not TABLE.match(lines[j][2])
            and not LIST.match(lines[j][2])
        ):
            j += 1
        ps, pe = lines[i][0], lines[j - 1][1]
        if j == i + 1 and pseudo_heading(text[ps:pe]):
            out.append({"t": "h", "s": ps, "e": pe, "lvl": 7, "title": clean_heading(text[ps:pe])})
        else:
            out.append({"t": "p", "s": ps, "e": pe})
        i = j
    return out


def tree(blocks):
    root = {"lvl": 0, "head": None, "own": [], "kids": []}
    stack = [root]
    for b in blocks:
        if b["t"] == "h":
            sec = {"lvl": b["lvl"], "head": b, "own": [], "kids": []}
            while stack[-1]["lvl"] >= b["lvl"]:
                stack.pop()
            stack[-1]["kids"].append(sec)
            stack.append(sec)
        else:
            stack[-1]["own"].append(b)
    return root


def span(sec):
    if sec["head"]:
        s = sec["head"]["s"]
    elif sec["own"]:
        s = sec["own"][0]["s"]
    elif sec["kids"]:
        s = span(sec["kids"][0])[0]
    else:
        return None, None
    if sec["kids"]:
        e = span(sec["kids"][-1])[1]
    elif sec["own"]:
        e = sec["own"][-1]["e"]
    else:
        e = sec["head"]["e"]
    return s, e


class Doc:
    def __init__(self, text):
        self.t = text
        self.cache = {}

    def tk(self, s, e):
        k = (s, e)
        if k not in self.cache:
            self.cache[k] = ntok(self.t[s:e])
        return self.cache[k]


def hard_split(D, s, e):
    enc = tok(D.t[s:e], add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    out = []
    i = 0
    while i < len(offs):
        j = min(len(offs), i + MAXT)
        if j < len(offs):
            k = j
            while k > i + MAXT // 2 and not D.t[s + offs[k][0] - 1 : s + offs[k][0]].isspace():
                k -= 1
            j = k if k > i + MAXT // 2 else j
        a, b = s + offs[i][0], s + offs[j - 1][1]
        out.append({"s": a, "e": b, "tok": j - i, "h": False})
        i = j
    return out


def units_of(D, b):
    """atomic units (each <= MAXT tokens) of one non-heading block"""
    s, e = b["s"], b["e"]
    t = D.tk(s, e)
    if t <= MAXT:
        return [{"s": s, "e": e, "tok": t, "h": False}]
    if b["t"] == "table":
        res = []
        hs, he = b["hdr"] if b["hdr"] else (None, None)
        htok = D.tk(hs, he) if b["hdr"] else 0
        rows = b["rows"]
        for k, (rs, re_) in enumerate(rows):
            st0 = hs if (k == 0 and b["hdr"]) else rs
            rt = D.tk(st0, re_)
            if rt > MAXT:
                res += hard_split(D, st0, re_)
            else:
                res.append(
                    {
                        "s": st0,
                        "e": re_,
                        "tok": rt,
                        "h": False,
                        "tbl": (hs, he, htok) if (b["hdr"] and k > 0 and htok <= MAXT // 3) else None,
                    }
                )
        return res
    if b["t"] == "list":
        res = []
        for a, z in b["items"]:
            it = D.tk(a, z)
            res += [{"s": a, "e": z, "tok": it, "h": False}] if it <= MAXT else sent_units(D, a, z)
        return res
    return sent_units(D, s, e)


def sent_units(D, s, e):
    res = []
    for sb in sentenize(D.t[s:e]):
        a, z = s + sb.start, s + sb.stop
        st_ = D.tk(a, z)
        res += [{"s": a, "e": z, "tok": st_, "h": False}] if st_ <= MAXT else hard_split(D, a, z)
    return res


def pack(units, path):
    pieces, cur, ct = [], [], 0

    def flush():
        nonlocal cur, ct
        if not cur:
            return
        carry = []
        while len(cur) > 1 and cur[-1]["h"]:
            carry.insert(0, cur.pop())
        tb = cur[0].get("tbl")
        pieces.append(
            {
                "s": cur[0]["s"],
                "e": cur[-1]["e"],
                "tok": sum(u["tok"] for u in cur) + (tb[2] if tb else 0),
                "path": path,
                "tbl": tb,
            }
        )
        cur = carry
        ct = sum(u["tok"] for u in cur)

    for u in units:
        if cur and not all(x["h"] for x in cur) and (ct + u["tok"] + 2 > MAXT or ct >= TARGET):
            flush()
        if not cur and u.get("tbl"):
            ct = u["tbl"][2]
        cur.append(u)
        ct += u["tok"] + 2
    flush()
    return pieces


def emit(D, sec, path):
    p = path + ([sec["head"]["title"]] if sec["head"] else [])
    s, e = span(sec)
    if s is None or e is None:
        return []
    if D.tk(s, e) <= MAXT:
        return [{"s": s, "e": e, "tok": D.tk(s, e), "path": p, "tbl": None}]
    units = (
        [{"s": sec["head"]["s"], "e": sec["head"]["e"], "tok": D.tk(sec["head"]["s"], sec["head"]["e"]), "h": True}]
        if sec["head"]
        else []
    )
    for b in sec["own"]:
        units += units_of(D, b)
    out = []
    if sec["kids"] and units and all(u["h"] for u in units):  # heading only: let it lead the first subsection
        kid_pieces = [x for k in sec["kids"] for x in emit(D, k, p)]
        if kid_pieces and kid_pieces[0]["tok"] + units[0]["tok"] <= MAXT and not kid_pieces[0]["tbl"]:
            kid_pieces[0] = dict(kid_pieces[0], s=units[0]["s"], tok=kid_pieces[0]["tok"] + units[0]["tok"])
            return kid_pieces
        return pack(units, p) + kid_pieces
    out += pack(units, p)
    for k in sec["kids"]:
        out += emit(D, k, p)
    return out


def common(a, b):
    r = []
    for x, y in zip(a, b):
        if x != y:
            break
        r.append(x)
    return r


def merge_small(pieces):
    out = []
    for p in pieces:
        if out and not p["tbl"] and (p["tok"] < MINT or out[-1]["tok"] < MINT) and out[-1]["tok"] + p["tok"] <= MAXT:
            q = out[-1]
            out[-1] = dict(
                q,
                e=p["e"],
                tok=q["tok"] + p["tok"],
                path=common(q["path"], p["path"]) if q["path"] != p["path"] else q["path"],
            )
        else:
            out.append(p)
    return out


def fmt_date(d):
    if d is None:
        return None
    d = str(d)
    if re.fullmatch(r"\d{9,11}", d):
        return time.strftime("%Y-%m-%d", time.gmtime(int(d)))
    return d[:10]


def header(m, path, title):
    lines = [title]
    src = org_label(m.get("site"))
    d = (m.get("dates") or {}).get("published") or (m.get("dates") or {}).get("main_published_at")
    dt = (m.get("semantic") or {}).get("doc_type") or m.get("doc_type")
    d = fmt_date(d)
    src_line = f"Источник: {src}" + (
        f"; дата: {d}"
        if d and (dt in NEWSY or m.get("kb_build") in ("nsu_social", "akademgorodok_media", "akademgorodok_news"))
        else ""
    )
    lines.append(src_line)
    pth = [x for x in path if norm_title(x) != norm_title(title) and x]
    if pth:
        lines.append("Раздел: " + " › ".join(pth))
    return lines


def fit_header(lines, body_tok):
    lines = list(lines)

    def ok(ls):
        h = "\n".join(ls) + "\n\n"
        return ntok(h) <= HDR_MAX and ntok(PREFIX + h) + body_tok <= EMB_MAX - 4

    while not ok(lines):
        if len(lines) > 2:
            segs = lines[2][len("Раздел: ") :].split(" › ")
            lines = lines[:2] + (["Раздел: " + " › ".join(segs[1:])] if len(segs) > 1 else [])
        elif len(lines[0]) > 80:
            lines[0] = lines[0][:77].rstrip() + "…"
        elif len(lines) > 1:
            lines = lines[:1]
        else:
            return ""
    return "\n".join(lines) + "\n\n"


def first_sentence(text):
    for ln in text.split("\n"):
        s = ln.strip(" #*-|")
        if len(s) >= 40 and not s.startswith(("Источник", "Дата")):
            m = re.match(r"(.{40,300}?[.!?])(\s|$)", s)
            return (m.group(1) if m else s[:300]).strip()
    return ""


def process(job):
    m, text = job
    D = Doc(text)
    title = re.sub(r"\s+", " ", (m.get("title") or "").strip()) or first_sentence(text)[:120] or m["canonical_url"]
    blocks = parse(text)
    if not blocks:
        return None
    pieces = []
    for p in merge_small(emit(D, tree(blocks), [])):
        bt = D.tk(p["s"], p["e"]) + (p["tbl"][2] if p["tbl"] else 0)
        if bt > MAXT + 40:  # token estimate drifted (many tiny blocks): cut exactly by token offsets
            pieces += [dict(h, path=p["path"], tbl=None) for h in hard_split(D, p["s"], p["e"])]
        else:
            pieces.append(p)
    # render (only when table continuations exist)
    rendered = any(p["tbl"] for p in pieces)
    chunks, parts, pos = [], [], 0
    for p in pieces:
        body = text[p["s"] : p["e"]]
        if p["tbl"]:
            body = text[p["tbl"][0] : p["tbl"][1]] + "\n" + body
        if rendered:
            st_ = pos
            parts.append(body)
            pos += len(body) + 2
            en_ = st_ + len(body)
        else:
            st_, en_ = p["s"], p["e"]
        bt = ntok(body)
        h = fit_header(header(m, p["path"], title), bt)
        if ntok(PREFIX + h + body) > EMB_MAX:  # rare: body itself too long after header -> keep, flagged
            flag = "over_limit"
        else:
            flag = None
        letters = sum(c.isalpha() for c in body)
        chunks.append(
            {
                "start_char": st_,
                "end_char": en_,
                "text": body,
                "index_text": h + body,
                "heading_path": p["path"],
                "tokens": bt,
                "tokens_index": ntok(PREFIX + h + body),
                "has_table": "|" in body[:2] or "\n|" in body,
                "table_header_repeated": bool(p["tbl"]),
                "flag": flag,
                "low_content": letters < 25,
                "source_spans": (
                    [{"start_char": p["tbl"][0], "end_char": p["tbl"][1], "role": "repeated_table_header"}]
                    if p["tbl"]
                    else []
                )
                + [{"start_char": p["s"], "end_char": p["e"], "role": "body"}],
                "source_joiner": "\n" if p["tbl"] else "",
            }
        )
    full = "\n\n".join(parts) if rendered else text
    return {
        "m": m,
        "title": title,
        "full": full,
        "rendered": rendered,
        "chunks": chunks,
        "doc_tokens": ntok(text) if len(text) < 400000 else int(len(text) / 3.4),
    }


def annotation(m, text):
    s = (m.get("semantic") or {}).get("summary_ru")
    if s:
        return s.strip()
    b = m.get("kb_build")
    if b in CARD_ANN:
        return CARD_ANN[b]
    if b and b.startswith("wikipedia"):
        return "Статья Википедии: " + first_sentence(text)
    return first_sentence(text)


def qscore(m):
    q = TIER_Q.get(m["priority"]["tier"], 0.6)
    if m.get("review_status") == "llm_reformatted_partial":
        q -= 0.05
    if (m.get("quality") or {}).get("ocr"):
        q -= 0.05
    if any(
        (f or {}).get("flag") in ("outdated", "dated_past_year", "superseded")
        for f in ((m.get("semantic") or {}).get("currentness_flags") or [])
    ):
        q -= 0.05
    return round(max(0.3, min(1.0, q)), 3)


def jobs():
    previous_key = None
    for d, m in paired_rows(Path(S)):
        key = (
            {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(m["priority"]["tier"], 9),
            m["kb_build"],
            m["canonical_url"],
            m["id"],
        )
        if previous_key is not None and key < previous_key:
            raise ValueError("Source export must be sorted before deterministic chunking")
        previous_key = key
        if m.get("duplicate_of") or (m.get("recommendation") or {}).get("action") == "exclude":
            continue
        if m.get("site") in SKIP_SITES or SKIP_TITLE.search(m.get("title") or ""):
            continue
        if len(d["text"].strip()) >= 80:
            yield (m, d["text"])


def main():
    os.makedirs(f"{OUT}/ragcore", exist_ok=True)
    os.makedirs(f"{OUT}/chunks", exist_ok=True)
    t0 = time.time()
    seen = {}
    st = collections.Counter()
    mapping = {}
    gi = 0
    doc_ids = set()
    chunk_ids = set()
    fc = None
    try:
        with (
            Pool(int(os.environ.get("WORKERS", "4")), initializer=init) as pool,
            open_gzip(Path(OUT) / "docs.jsonl.gz") as fd,
            open_gzip(Path(OUT) / "ragcore/chunked_texts_about_nsu_with_metadata.jsonl.gz") as fr,
        ):
            for r in pool.imap(process, jobs(), chunksize=2):
                if r is None:
                    continue
                m = r["m"]
                sem = m.get("semantic") or {}
                di = st["docs"]
                if m["id"] in doc_ids:
                    raise ValueError("Duplicate document ID at output")
                doc_ids.add(m["id"])
                binding = {
                    "source_version_id": m["kb_version_id"],
                    "source_text_sha256": m["text_sha256"],
                    "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                }
                urls = (
                    [m["canonical_url"]]
                    if not (m.get("provenance") or {}).get("sources")
                    else [x.get("url") for x in m["provenance"]["sources"] if isinstance(x, dict) and x.get("url")][:8]
                    or [m["canonical_url"]]
                )
                date = fmt_date(
                    (m.get("dates") or {}).get("published") or (m.get("dates") or {}).get("main_published_at")
                )
                ann = annotation(m, r["full"])
                q = qscore(m)
                rc = []
                ids = []
                stem = f"{m['id']}#v{m['kb_version_id']}#c"
                for ci, c in enumerate(r["chunks"]):
                    if st["chunks"] % 50000 == 0:
                        if fc is not None:
                            fc.close()
                        fc = open_gzip(Path(OUT) / f"chunks/part-{st['chunks'] // 50000:05d}.jsonl.gz")
                    cid = f"{stem}{ci:03d}"
                    ids.append(cid)
                    if cid in chunk_ids:
                        raise ValueError("Duplicate chunk ID")
                    chunk_ids.add(cid)
                    key = hashlib.sha256(re.sub(r"\s+", " ", c["text"].lower()).strip().encode()).hexdigest()
                    dup = seen.get(key)
                    capped = ci >= DOC_CAP and m["priority"]["tier"] != "P0"
                    indexed = not dup and not c["low_content"] and not capped
                    if indexed:
                        seen[key] = cid
                    if c["flag"] == "over_limit":
                        raise ValueError("FRIDA token ceiling exceeded: " + cid)
                    st["doc_cap"] += capped
                    st["chunks"] += 1
                    st["dup"] += bool(dup)
                    st["low_content"] += c["low_content"]
                    st["over_limit"] += c["flag"] == "over_limit"
                    st["table_hdr_repeated"] += c["table_header_repeated"]
                    rec = {
                        "chunk_id": cid,
                        "doc_id": m["id"],
                        "kb_build": m["kb_build"],
                        "chunk_index": ci,
                        "n_chunks": len(r["chunks"]),
                        **binding,
                        "start_char": c["start_char"],
                        "end_char": c["end_char"],
                        "text": c["text"],
                        "index_text": c["index_text"],
                        "source_spans": c["source_spans"],
                        "source_joiner": c["source_joiner"],
                        "heading_path": c["heading_path"],
                        "tokens_frida": c["tokens"],
                        "tokens_index_frida": c["tokens_index"],
                        "has_table": c["has_table"],
                        "table_header_repeated": c["table_header_repeated"],
                        "indexed": indexed,
                        "not_indexed_reason": None
                        if indexed
                        else ("duplicate_chunk" if dup else "low_content" if c["low_content"] else "doc_chunk_cap"),
                        "dup_of": dup,
                        "low_content": c["low_content"],
                        "title": r["title"],
                        "url": urls[0],
                        "site": m.get("site"),
                        "source_label": org_label(m.get("site")),
                        "tier": m["priority"]["tier"],
                        "doc_type": sem.get("doc_type") or m.get("doc_type"),
                        "topics": sem.get("topics") or [],
                        "date": date,
                        "language": m.get("language", "ru"),
                        "quality_score": q,
                        "prev_chunk_id": f"{stem}{ci - 1:03d}" if ci else None,
                        "next_chunk_id": f"{stem}{ci + 1:03d}" if ci + 1 < len(r["chunks"]) else None,
                    }
                    fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if indexed:
                        mapping[str(gi)] = {"doc_index": di, "local_chunk_index": len(rc)}
                        gi += 1
                        rc.append(
                            {
                                "start_char": c["start_char"],
                                "end_char": c["end_char"],
                                "num_tokens": c["tokens"],
                                "chunk_id": cid,
                            }
                        )
                st["docs"] += 1
                st["rendered_docs"] += r["rendered"]
                fd.write(
                    json.dumps(
                        {
                            "doc_id": m["id"],
                            "kb_build": m["kb_build"],
                            "title": r["title"],
                            "annotation": ann,
                            "urls": urls,
                            "tier": m["priority"]["tier"],
                            **binding,
                            "doc_type": sem.get("doc_type") or m.get("doc_type"),
                            "quality_score": q,
                            "n_chunks": len(r["chunks"]),
                            "n_indexed": len(rc),
                            "doc_tokens_frida": r["doc_tokens"],
                            "text_rendered": r["rendered"],
                            "chunk_ids": ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fr.write(
                    json.dumps(
                        {
                            "content_source": m["kb_build"],
                            "kb_id": m["id"],
                            **binding,
                            "url": urls if len(urls) > 1 else urls[0],
                            "doc_title": r["title"],
                            "doc_annotation": ann,
                            "doc_full_text": r["full"],
                            "chunks": rc,
                            "num_tokens": r["doc_tokens"],
                            "quality_score": q,
                            "tier": m["priority"]["tier"],
                            "doc_type": sem.get("doc_type") or m.get("doc_type"),
                            "date": date,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if st["docs"] % 1000 == 0:
                    print(
                        "chunked",
                        st["docs"],
                        "documents",
                        st["chunks"],
                        "chunks",
                        round(time.time() - t0),
                        "s",
                        flush=True,
                    )
    finally:
        if fc is not None:
            fc.close()
    with open_gzip(Path(OUT) / "ragcore/chunk_mapping_to_texts.json.gz") as f:
        json.dump(mapping, f)
    st["indexed"] = gi
    result = {
        **dict(st),
        "unique_document_ids": len(doc_ids),
        "unique_chunk_ids": len(chunk_ids),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "tokenizer_snapshot": Path(TOKENIZER_PATH).name,
        "seconds": round(time.time() - t0, 2),
    }
    Path(OUT, "stats.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
