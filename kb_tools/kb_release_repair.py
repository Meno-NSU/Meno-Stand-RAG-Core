"""Repair a pinned KB export, preserving source texts and an explicit change ledger."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re

from kb_mailto import decode_mailto

NEWS_HEADING = re.compile(r"(?m)^#{1,6}\s+(?:Последние новости|Читайте также|Другие новости|Похожие новости)\s*$")
TEASERS = (
    "Очередная лекция из осеннего цикла социального проекта собрала почти 100 слушателей.",
    "В городское хозяйство Искитима будут внедрять",
    "НГУ стал одним из участников альянса по цифровизации",
)
TIER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def open_gzip(path):
    """No timestamp or filename in the gzip envelope."""
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    return io.TextIOWrapper(gzip.GzipFile(filename=str(path), mode="wb", mtime=0), encoding="utf-8")


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        yield from map(json.loads, source)


def paired_rows(root):
    seen = set()
    for text_path in sorted((root / "texts").glob("*.jsonl.gz")):
        meta_path = root / "meta" / text_path.name
        for text, meta in zip_longest(rows(text_path), rows(meta_path)):
            if text is None or meta is None or text["id"] != meta["id"]:
                raise ValueError(f"unpaired source records in {text_path.name}")
            if text["id"] in seen:
                raise ValueError("duplicate source ID: " + text["id"])
            seen.add(text["id"])
            if digest(text["text"]) != meta["text_sha256"]:
                raise ValueError("source hash mismatch: " + text["id"])
            yield text, meta


def trim_news_tail(text, site, approved_tail_hashes=None):
    """Require a template heading AND multiple known, repeated sidebar teasers."""
    if (site or "").removeprefix("www.") != "nsu.ru":
        return text, None
    for match in NEWS_HEADING.finditer(text):
        tail = text[match.start() :]
        confirmed = sum(marker in tail for marker in TEASERS) >= 2 or digest(tail) in (approved_tail_hashes or set())
        if len(tail) <= 20000 and confirmed:
            kept = text[: match.start()].rstrip()
            if not kept:
                return text, None
            return kept + "\n", {"start_char": match.start(), "removed_text": tail}
    return text, None


def mail_evidence(originals):
    """Extract only explicit, unambiguous address encodings from hash-bound HTML."""
    from bs4 import BeautifulSoup

    mappings = defaultdict(list)
    valid = defaultdict(list)
    unresolved = []
    for path in sorted(originals.glob("*.html*")):
        sha = path.name.split(".")[0]
        if not re.fullmatch("[a-f0-9]{64}", sha):
            continue
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("original HTML hash mismatch: " + str(path))
        soup = BeautifulSoup(raw, "lxml")
        for anchor in soup.find_all("a", href=lambda x: x and x.startswith("mailto:")):
            href = anchor["href"]
            address = decode_mailto(href)
            visible = anchor.get_text("", strip=True).strip()
            record = {"original_id": sha, "href": href, "visible_text": visible, "email": address}
            if address:
                valid[address].append(record)
                # Exclude labels such as "write to us", domain-only anchors, and ambiguous encodings.
                if visible and "@" not in visible and visible == address.replace("@", ""):
                    mappings[visible].append(record)
            elif visible and "@" not in visible and "@." in href:
                unresolved.append(record)
    # A malformed mailto can be corrected only if another saved original explicitly corroborates it.
    for record in unresolved:
        candidate = record["href"][7:].replace("@.", "@", 1)
        if candidate in valid and record["visible_text"] == candidate.replace("@", ""):
            mappings[record["visible_text"]].append(
                {**record, "email": candidate, "corroboration": valid[candidate][:3]}
            )
    return {old: evidence for old, evidence in mappings.items() if len({r["email"] for r in evidence}) == 1}, unresolved


def repair_emails(text, meta, mappings):
    original_id = ((meta.get("provenance") or {}).get("primary") or {}).get("original_id")
    changes = []
    for old, evidence in mappings.items():
        if old not in text and old.replace("_", "\\_") not in text:
            continue
        # Domain-looking tokens can be real websites. Outside their own source HTML,
        # accept them only in explicitly named email cells or email-labelled lines.
        own_evidence = [r for r in evidence if r["original_id"] == original_id]
        new = evidence[0]["email"]
        variants = sorted({old, old.replace("_", "\\_")}, key=len, reverse=True)
        for variant in variants:
            pattern = re.compile(r"(?<![\w@/.:?=&%+\-])" + re.escape(variant) + r"(?![\w@/.:?=&%+\-])")
            updated = []
            pos = 0
            for match in pattern.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line = text[line_start : line_end if line_end >= 0 else len(text)]
                prefix = text[max(0, line_start - 500) : match.start()]
                email_context = bool(re.search(r"e-?mail|электронн\w* почт|эл\.\s*почт", prefix, re.I))
                # Explicitly verified contacts-page aliases are also safe in a table
                # whose preceding header identifies the mail column.
                table_email = line.startswith("|") and "Электронная почта" in text[:line_start]
                if not own_evidence and not email_context and not table_email:
                    continue
                updated.extend([text[pos : match.start()], new])
                pos = match.end()
                changes.append({"old": variant, "new": new, "evidence": (own_evidence or evidence)[:3]})
            if pos:
                text = "".join(updated) + text[pos:]
    return text, changes


def sort_key(record):
    _, meta = record
    return (TIER.get((meta.get("priority") or {}).get("tier"), 9), meta["kb_build"], meta["canonical_url"], meta["id"])


def update_duplicates(records):
    groups = defaultdict(list)
    changes = []
    for text, meta in records:
        groups[meta["text_sha256"]].append(meta)
    for group in groups.values():
        canonical = min(
            group,
            key=lambda m: (
                (m.get("recommendation") or {}).get("action") == "exclude"
                and (m.get("recommendation") or {}).get("reason") != "duplicate_content",
                TIER.get((m.get("priority") or {}).get("tier"), 9),
                m["kb_build"],
                m["canonical_url"],
                m["id"],
            ),
        )
        for meta in group:
            before = meta.get("duplicate_of")
            recommendation = meta.get("recommendation") or {}
            after = canonical["id"] if meta is not canonical else None
            if after:
                meta["duplicate_of"] = after
                if recommendation.get("action") != "exclude" or recommendation.get("reason") == "duplicate_content":
                    meta["recommendation"] = {"action": "exclude", "reason": "duplicate_content"}
            else:
                meta.pop("duplicate_of", None)
                if recommendation.get("reason") == "duplicate_content":
                    meta["recommendation"] = {"action": "keep", "reason": None}
            if before != after:
                changes.append({"id": meta["id"], "old_duplicate_of": before, "new_duplicate_of": after})
                meta.setdefault("processing", {})["duplicate_check"] = "exact_sha256_after_repair"
    return changes


def repair(source, output, originals, source_revision):
    now = datetime.now(timezone.utc).isoformat()
    mappings, unresolved = mail_evidence(originals)
    records = list(paired_rows(source))
    # Other crawler dates show different recommendations. Approve these only
    # when the complete suffix repeats across >=5 pages and contains >=2 dates.
    suffixes = Counter()
    for text_row, meta in records:
        if (meta.get("site") or "").removeprefix("www.") != "nsu.ru":
            continue
        for match in NEWS_HEADING.finditer(text_row["text"]):
            tail = text_row["text"][match.start() :]
            if len(tail) <= 20000 and len(re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", tail)) >= 2:
                suffixes[digest(tail)] += 1
    approved_tail_hashes = {sha for sha, count in suffixes.items() if count >= 5}
    counts = Counter()
    history = output / "curation/kb_repair_2026-11-02"
    history.mkdir(parents=True, exist_ok=True)
    with (
        open_gzip(history / "previous_versions.jsonl.gz") as previous,
        open_gzip(history / "decisions.jsonl.gz") as decisions,
        open_gzip(history / "revisions.jsonl.gz") as revisions,
    ):
        for text_row, meta in records:
            before = text_row["text"]
            after, tail = trim_news_tail(before, meta.get("site"), approved_tail_hashes)
            after, mail_changes = repair_emails(after, meta, mappings)
            if after == before:
                continue
            previous.write(json.dumps({"text": text_row, "meta": meta}, ensure_ascii=False) + "\n")
            old_sha, old_version = meta["text_sha256"], meta["kb_version_id"]
            new_sha = digest(after)
            new_version = digest(meta["id"] + "\0" + new_sha)[:32]
            changes = []
            if tail:
                counts["news_tail_documents"] += 1
                counts["news_tail_characters_removed"] += len(tail["removed_text"])
                changes.append({"op": "remove_related_news_sidebar", **tail})
                if len(after.strip()) < 80:
                    meta["recommendation"] = {"action": "exclude", "reason": "empty_or_stub_after_sidebar_removal"}
                    counts["sidebar_only_stubs_excluded"] += 1
            if mail_changes:
                counts["email_documents"] += 1
                counts["email_replacements"] += len(mail_changes)
                changes.append({"op": "decode_verified_mailto", "changes": mail_changes})
                meta.setdefault("processing", {})["text_is_verbatim_subset_of_page"] = False
                meta["processing"]["email_decoding_verified_against_saved_html"] = True
            meta.setdefault("edit_log", []).append(
                {
                    "op": "evidence_bound_kb_repair",
                    "at": now,
                    "by": "codex",
                    "source_revision": source_revision,
                    "prev_text_sha256": old_sha,
                    "prev_kb_version_id": old_version,
                    "changes": changes,
                }
            )
            text_row["text"] = after
            meta["text_sha256"] = new_sha
            meta["kb_version_id"] = new_version
            meta.setdefault("quality", {})["chars"] = len(after)
            meta["quality"]["post_repair_hash_verified"] = True
            # Do not carry unsupported sidebar evidence into current semantic metadata.
            semantic = meta.get("semantic") or {}
            removed_flags = []
            flags = []
            for flag in semantic.get("currentness_flags") or []:
                evidence = flag.get("evidence") if isinstance(flag, dict) else None
                if tail and evidence and evidence in tail["removed_text"] and evidence not in after:
                    removed_flags.append(flag)
                else:
                    flags.append(flag)
            if removed_flags:
                semantic["currentness_flags"] = flags
                meta["edit_log"][-1]["removed_sidebar_metadata_flags"] = removed_flags
            if mail_changes:
                quality_issues = semantic.get("quality_issues") or []
                semantic["quality_issues"] = [
                    issue
                    for issue in quality_issues
                    if not ("Email values lack the '@' sign" in str(issue) and "priemnsu.ru" not in after)
                ]
            decision = {
                "decision_id": "repair-" + digest(meta["id"] + old_sha + new_sha),
                "document_id": meta["id"],
                "version_id": old_version,
                "expected_text_sha256": old_sha,
                "action": "needs_fix",
                "reason": [c["op"] for c in changes],
                "reviewer": "codex",
                "reviewed_at": now,
                "evidence_urls": [meta["canonical_url"]],
                "evidence_pages": [],
            }
            decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
            revisions.write(
                json.dumps(
                    {
                        **decision,
                        "old_version_id": old_version,
                        "new_version_id": new_version,
                        "text_sha256": new_sha,
                        "text": after,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts["changed_text_documents"] += 1
    dedup_changes = update_duplicates(records)
    records.sort(key=sort_key)
    parts = []
    for offset in range(0, len(records), 2000):
        name = f"part-{offset // 2000:05d}.jsonl.gz"
        text_path, meta_path = output / "kb/sources/v1/texts" / name, output / "kb/sources/v1/meta" / name
        batch = records[offset : offset + 2000]
        with open_gzip(text_path) as tf, open_gzip(meta_path) as mf:
            for text, meta in batch:
                tf.write(json.dumps(text, ensure_ascii=False) + "\n")
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
        parts.append(
            {
                "name": name,
                "records": len(batch),
                "texts_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
                "meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            }
        )
    count_field = lambda fn: dict(Counter(fn(m) for _, m in records))
    manifest = {
        "name": "NSU knowledge base — sources layer",
        "version": "v2.2",
        "built_at": now,
        "parent_repo_revision": source_revision,
        "records": len(records),
        "chars": sum(len(t["text"]) for t, _ in records),
        "by_tier": count_field(lambda m: m["priority"]["tier"]),
        "by_build": count_field(lambda m: m["kb_build"]),
        "by_review_status": count_field(lambda m: m["review_status"]),
        "by_doc_type": count_field(lambda m: (m.get("semantic") or {}).get("doc_type") or m.get("doc_type")),
        "duplicates_flagged": sum(bool(m.get("duplicate_of")) for _, m in records),
        "recommended_exclude": sum((m.get("recommendation") or {}).get("action") == "exclude" for _, m in records),
        "repair_counts": dict(counts),
        "parts": parts,
        "changed_kb_version_id_scheme": "first 32 hex chars of SHA256(id + NUL + text_sha256)",
        "join": "texts and meta parts have identical unique ids in identical order; join on id",
    }
    write_json(output / "kb/sources/v1/manifest.json", manifest)
    write_json(history / "duplicate_decisions.json", dedup_changes)
    write_json(
        history / "approved_sidebar_hashes.json",
        {"minimum_repeat_count": 5, "sha256_counts": {sha: suffixes[sha] for sha in sorted(approved_tail_hashes)}},
    )
    write_json(history / "email_evidence.json", {"mappings": mappings, "invalid_original_mailtos": unresolved})
    write_json(
        output / "repair_progress.json",
        {
            "stage": "sources_repaired",
            "source_revision": source_revision,
            "counts": dict(counts),
            "duplicate_pointer_changes": len(dedup_changes),
        },
    )
    print(
        json.dumps({"counts": dict(counts), "duplicate_pointer_changes": len(dedup_changes)}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--originals", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    args = p.parse_args()
    repair(args.source, args.output, args.originals, args.source_revision)
