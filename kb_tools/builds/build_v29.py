"""KB release from the HF v2.9 source layer (kb/releases/v2.9-social-places-2026-11-05/source_snapshot) plus selected
posts of the 2026-11-05 public social crawl (external/social/nsu_akadem_2026-11-05), chunked in one run by the
kb/chunks/v1 release's own tools/chunk_kb.py and indexed by its tools/rag_bm25.py.

usage: KB_FRIDA_TOKENIZER=<frida> WORKERS=16 python build_v29.py TOOLS SNAPSHOT CRAWL FEED_NAMES NEW_ROOT HF_REV

Layer records: the chunker's own rules (duplicate_of, recommended exclude, table.nsu.ru, external catalogues, < 80 chars),
except posts whose only exclusion reason is 'historical_timed_opportunity_requires_revalidation': they are kept (no
information loss; their text already states the publication date and that currentness is not confirmed).
Social labels: the layer stores bare 't.me' / 'vk.com' as site, which the release's org_label turns into "official NSU
channel" for every community; the site is taken from the post URL instead and the community name from the crawl
registry; the official NSU label stays only for t.me/nsuniversity and the NSU VK community (id 37080997).
FIT student review cards (nsu_student_reviews): as in the earlier derived release (build_fitrev.py) — title "Отзывы
студентов ФИТ о преподавателе: <name>", label "анонимные отзывы студентов ФИТ (не официальные сведения НГУ)", tier P1,
doc_type student_review_card, URL kb://student_reviews/fit/<card_id> (the layer links them to the private HF dataset,
a dead link for users), machine provenance removed (DOCX paragraph numbers, SHA-256 hashes); card text otherwise kept.
Crawl posts: communities whose relation_scope is in SCOPES, URL not already in the layer, body (letters/digits key) not
repeating a layer or earlier crawl post (originals first, then oldest first), body with >= 25 letters.
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

TOOLS, SNAP, CRAWL, NAMES, NEW = (Path(a) for a in sys.argv[1:6])
HF_REV = sys.argv[6]
NV = NEW / "kb/chunks/v1"
SCOPES = ("nsu_related_public_description", "akadem_related_or_mapped_business", "academic_research_network")
KEEP_EXCLUDE = "historical_timed_opportunity_requires_revalidation"
CRAWL_BUILD = "nsu_akadem_social_2026_11_05"
OFFICIAL_TG, OFFICIAL_VK = "nsuniversity", "37080997"
SKIP_SITES = {"table.nsu.ru"}  # verbatim from tools/chunk_kb.py
SKIP_TITLE = re.compile(r"Атлас новых исследований на основе EconLit|^Перечень рецензируемых научных изданий", re.I)
MENTION = re.compile(r"\[(?:club|public|id|event)\d+\|([^\]\n]+)\]")
CAVEAT = re.compile(r"\AПубликация (?:Telegram|VK)[^\n]*\n+")
TG_URL = re.compile(r"https?://t\.me/(?:s/)?([A-Za-z0-9_]+)")
VK_WALL = re.compile(r"https?://(?:m\.)?vk\.com/.*?wall(-?\d+)_\d+")
VK_DOC = re.compile(r"https?://(?:m\.)?vk\.com/doc-?\d+")
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()  # noqa: E731


def body_key(text):
    body = MENTION.sub(r"\1", CAVEAT.sub("", text))
    return sha(re.sub(r"[\W_]+", "", body.lower()))


REV_SITE, REV_LABEL = "student_reviews.fit", "анонимные отзывы студентов ФИТ (не официальные сведения НГУ)"


def clean_card(md, name):  # verbatim rules of build_fitrev.clean
    md = re.sub(r"\A# .*\n", f"# Отзывы студентов ФИТ о преподавателе: {name}\n", md)
    md = re.sub(r"[;,]? ?абзацы \d+(?:, \d+)*", "", md)
    md = re.sub(r" ?Фрагменты оригинала: \d+(?:, \d+)*\.", "", md)
    md = re.sub(r" ?SHA-256: `[0-9a-f]{64}`\.", "", md)
    md = re.sub(r" ?Подтверждение: word/[\w./:;, ]+\.", "", md)
    md = re.sub(r"\n\nХэш сохранённого официального HTML: `[0-9a-f]{64}`\.", "", md)
    md = re.sub(r"Номера абзацев относятся к соответствующему XML-файлу DOCX, не к страницам Word\. ?", "", md)
    return md.strip() + "\n"


def title_of(body):
    for line in body.splitlines():
        line = re.sub(r"\s+", " ", line.lstrip("#").strip())
        if re.search(r"\w", line):
            return line if len(line) <= 110 else line[:110].rstrip() + "…"
    return ""


def main():
    t0 = time.time()
    if NEW.exists():
        sys.exit(f"{NEW} exists")
    sys.path.insert(0, str(TOOLS))
    from kb_release_repair import paired_rows

    reg = {c["source_id"]: c for c in map(json.loads, gzip.open(CRAWL / "communities.jsonl.gz", "rt", encoding="utf-8"))}
    tg_name = {k.split(":", 1)[1].lower(): v for k, v in json.loads(NAMES.read_text(encoding="utf-8")).items()
               if k.startswith("telegram:")}
    vk_by_id = {}
    for c in reg.values():
        if c["platform"] == "telegram" and c.get("handle") and c.get("title"):
            tg_name.setdefault(c["handle"].lower(), c["title"].strip())
        elif c["platform"] == "vk":
            vk_by_id[str(c["provider_id"]).lstrip("-")] = (c.get("handle") or f"club{c['provider_id']}", (c.get("title") or "").strip())
    labels = {}

    def site_of(url, plat=None):
        m = TG_URL.match(url or "")
        if m and plat in (None, "telegram"):
            slug = m.group(1).lower()
            site = f"t.me/{slug}"
            if slug == OFFICIAL_TG:
                labels[site] = "НГУ, официальный Telegram-канал"  # the release's own label: index_text unchanged
            else:
                name = tg_name.get(slug)
                labels[site] = f"Telegram-канал «{name}» (t.me/{slug})" if name else f"Telegram-канал t.me/{slug}"
            return site
        m = VK_WALL.match(url or "")
        if m:
            oid = m.group(1).lstrip("-")
            if oid == OFFICIAL_VK:
                labels["vk.com/nsu"] = "НГУ, официальное сообщество ВКонтакте"
                return "vk.com/nsu"
            handle, name = vk_by_id.get(oid, (f"club{oid}" if m.group(1).startswith("-") else f"id{oid}", ""))
            site = f"vk.com/{handle}"
            labels[site] = f"сообщество ВКонтакте «{name}» (vk.com/{handle})" if name else f"сообщество ВКонтакте vk.com/{handle}"
            return site
        if VK_DOC.match(url or ""):
            labels["vk.com/doc"] = "документ, опубликованный во ВКонтакте"
            return "vk.com/doc"
        return None

    # ---- layer records
    jobs, st = [], {"layer_records": 0, "layer_taken": 0, "kept_historical": 0, "skipped": {}, "relabelled_social": 0}
    seen_url, seen_body = set(), set()
    for t, m in paired_rows(SNAP):
        st["layer_records"] += 1
        text = t["text"]
        seen_url.add(m.get("canonical_url"))
        social = m.get("site") in ("t.me", "vk.com") or (m.get("semantic") or {}).get("doc_type") == "social_post"
        if social:
            seen_body.add(body_key(text))
        rec = m.get("recommendation") or {}
        reasons = rec.get("reasons") if isinstance(rec.get("reasons"), list) else [rec.get("reason")]
        why = None
        if m.get("duplicate_of"):
            why = "duplicate_of"
        elif rec.get("action") == "exclude":
            if set(map(str, reasons)) == {KEEP_EXCLUDE}:
                st["kept_historical"] += 1
            else:
                why = "recommended_exclude"
        if why is None and (m.get("site") in SKIP_SITES or SKIP_TITLE.search(m.get("title") or "")):
            why = "skip_site_or_catalogue"
        if why is None and len(text.strip()) < 80:
            why = "short_text"
        if why:
            st["skipped"][why] = st["skipped"].get(why, 0) + 1
            continue
        if m.get("site") in ("t.me", "vk.com"):
            site = site_of(m["canonical_url"])
            if site:
                m = {**m, "site": site}
                st["relabelled_social"] += 1
        if m.get("kb_build") == "nsu_student_reviews":
            name = re.match(r"\A# (.+)", text).group(1).strip()
            card = re.search(r"(fit-review-[0-9a-f]+)", m["canonical_url"]).group(1)
            text = clean_card(text, name)
            ts = sha(text)
            labels[REV_SITE] = REV_LABEL
            m = {**m, "title": f"Отзывы студентов ФИТ о преподавателе: {name}", "site": REV_SITE,
                 "canonical_url": f"kb://student_reviews/fit/{card}", "provenance": {},
                 "priority": {**m["priority"], "tier": "P1"}, "text_sha256": ts, "kb_version_id": sha(m["id"] + "\0" + ts)[:32],
                 "semantic": {**(m.get("semantic") or {}), "doc_type": "student_review_card"}}
            st["review_cards"] = st.get("review_cards", 0) + 1
        if rec.get("action") == "exclude":  # kept historical post: the chunker itself would skip it
            m = {**m, "recommendation": {**rec, "action": "keep", "kept_by": "build_v29: historical post, dated text"}}
        jobs.append((m, text))
    st["layer_taken"] = len(jobs)
    print(json.dumps(st, ensure_ascii=False), round(time.time() - t0), "s", flush=True)

    # ---- crawl posts
    cst = {"posts": 0, "scope_out": 0, "url_in_layer": 0, "short": 0, "dup_body": 0, "taken": 0, "by_scope": {}}
    files = {p: CRAWL / f"{p}.jsonl.gz" for p in ("telegram", "vk")}
    posts = []
    for p, f in files.items():
        for line in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(line)
            cst["posts"] += 1
            sc = (reg.get(r["source_id"]) or {}).get("relation_scope")
            if sc not in SCOPES:
                cst["scope_out"] += 1
                continue
            if r["url"] in seen_url:
                cst["url_in_layer"] += 1
                continue
            r["_scope"] = sc
            posts.append(r)
    posts.sort(key=lambda r: (bool(r.get("repost_sources")), r.get("published_at") or "9999", r["url"]))
    for r in posts:
        body = MENTION.sub(r"\1", r.get("text") or "").strip()
        if len(re.findall(r"[^\W\d_]", body)) < 25:
            cst["short"] += 1
            continue
        k = body_key(body)
        if k in seen_body:
            cst["dup_body"] += 1
            continue
        seen_body.add(k)
        seen_url.add(r["url"])
        tg = r["platform"] == "telegram"
        site = site_of(r["url"], r["platform"])
        if not site:
            continue
        date = (r.get("published_at") or "")[:10]
        caveat = (f"Публикация {'Telegram' if tg else 'VK'} от {date}." if date else
                  f"Публикация {'Telegram' if tg else 'VK'}, дата не указана.") + \
            " Сведения отражают момент публикации; актуальность объявления после этой даты не подтверждена."
        text = f"{caveat}\n\n{body}\n"
        doc_id = "kb-" + sha(f"{CRAWL_BUILD}:{r['url']}")[:32]
        ts = sha(text)
        m = {"id": doc_id, "kb_version_id": sha(doc_id + "\0" + ts)[:32], "text_sha256": ts,
             "title": title_of(body) or labels.get(site, site), "site": site, "kb_build": CRAWL_BUILD,
             "canonical_url": r["url"], "priority": {"tier": "P3", "reason": "public social caption, " + r["_scope"]},
             "dates": {"main_published_at": r.get("published_at"), "retrieved_at": r.get("retrieved_at")},
             "semantic": {"doc_type": "social_post"}, "language": "ru", "review_status": "unreviewed_public_caption"}
        jobs.append((m, text))
        cst["taken"] += 1
        cst["by_scope"][r["_scope"]] = cst["by_scope"].get(r["_scope"], 0) + 1
    print(json.dumps(cst, ensure_ascii=False), round(time.time() - t0), "s", flush=True)

    tier = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    jobs.sort(key=lambda j: (tier.get(j[0]["priority"]["tier"], 9), j[0]["kb_build"], j[0]["canonical_url"], j[0]["id"]))

    # ---- source binding: what was taken and how (its sha256 is the chunks' source_manifest_sha256)
    src_root = NEW / "sources"
    src_root.mkdir(parents=True)
    manifest = {
        "dataset": "Meno-NSU/NSU-Data-For-Meno", "revision": HF_REV,
        "layer": {"path": "kb/releases/v2.9-social-places-2026-11-05/source_snapshot",
                  "manifest_sha256": hashlib.sha256((SNAP / "manifest.json").read_bytes()).hexdigest(), **st},
        "crawl": {"path": "external/social/nsu_akadem_2026-11-05", "scopes": list(SCOPES),
                  "files": {p: hashlib.sha256(f.read_bytes()).hexdigest() for p, f in files.items()}, **cst},
        "selection": __doc__.split("\n\n", 2)[2].strip(),
    }
    (src_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    os.environ["KB_SOURCE_ROOT"] = str(src_root)
    import chunk_kb  # reads KB_SOURCE_ROOT/manifest.json and KB_FRIDA_TOKENIZER at import
    import rag_bm25

    base_label = chunk_kb.org_label
    chunk_kb.org_label = lambda site: labels.get(site) or base_label(site)
    chunk_kb.jobs = lambda: iter(jobs)
    chunk_kb.OUT = str(NV)
    NV.mkdir(parents=True)
    chunk_kb.main()
    shutil.copytree(TOOLS, NV / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(src_root / "manifest.json", NV / "source_selection.json")
    stats = json.loads((NV / "stats.json").read_text())
    stats["selection"] = {"layer": st, "crawl": cst}
    (NV / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    (NEW / "REVISION").write_text(f"{HF_REV}:v2.9+social-places-2026-11-05+social-crawl-selected\n")
    (NV / "README.md").write_text(
        "# Выпуск v2.9 + отобранные посты публичного краула 2026-11-05\n\n"
        f"Слой источников v2.9+social-places-2026-11-05 (HF {HF_REV}) и посты из external/social/nsu_akadem_2026-11-05, "
        "нарезанные одним запуском `tools/chunk_kb.py`; отбор описан в `source_selection.json`, BM25 — `tools/rag_bm25.py`.\n")
    print("chunked", round(time.time() - t0), "s", flush=True)
    rag_bm25.build(NV, int(os.environ.get("WORKERS", "8")))
    print("ALL", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
