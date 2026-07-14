#!/usr/bin/env python3
"""Download the Meno RAG stand resources from Yandex Disk.

The knowledge base is published as several independent public resources (a mix
of single files and one folder). Each is fetched to its final location under the
output directory; the corpus is saved under the name the backend expects
(``chunked_texts_about_nsu_with_metadata.jsonl``) even though it is published as
``..._and_scores.jsonl``.

``abbreviations.json`` has not been regenerated yet, so it is still pulled (as a
single small file) from the legacy combined archive. Replace its entry in
``RESOURCES`` with a dedicated link once the updated dictionary is published.
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

DOWNLOAD_API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

# Legacy combined archive, kept only as the (temporary) source of the unchanged
# abbreviations dictionary until a dedicated link exists.
LEGACY_ARCHIVE_URL = "https://disk.yandex.ru/d/eklv6Scj9OpbmQ"

BM25_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "params.index.json",
    "vocab.index.json",
)


@dataclass(frozen=True)
class Resource:
    label: str  # stable name for --only and logs
    public_url: str  # Yandex Disk public link
    path: str | None  # sub-path inside a public folder, or None for a directly published resource
    target: str  # destination path relative to the output directory
    kind: str  # "file" (raw download) or "archive" (folder published as a zip -> extract)


# Order matters only for readability; each resource is fetched independently.
RESOURCES: tuple[Resource, ...] = (
    Resource(
        label="corpus",
        public_url="https://disk.yandex.ru/d/4HUd56_sqwsoYA",
        path=None,
        # Published as ..._and_scores.jsonl; saved under the name the backend loads.
        target="chunked_texts_about_nsu_with_metadata.jsonl",
        kind="file",
    ),
    Resource(
        label="chunk_mapping",
        public_url="https://disk.yandex.ru/i/hRePe7eF8R66fQ",
        path=None,
        target="chunk_mapping_to_texts.json",
        kind="file",
    ),
    Resource(
        label="faiss_index",
        public_url="https://disk.yandex.ru/d/z6TaXeiflR7RtQ",
        path=None,
        target="knowledge/faiss_frida.index",
        kind="file",
    ),
    Resource(
        label="bm25",
        public_url="https://disk.yandex.ru/d/C7U5Hot_ktuWRw",
        path=None,
        target="knowledge/bm25",
        kind="archive",
    ),
    Resource(
        label="abbreviations",
        # TEMPORARY: pulled from the legacy archive until an updated file is published.
        public_url=LEGACY_ARCHIVE_URL,
        path="/knowledge/abbreviations.json",
        target="abbreviations.json",
        kind="file",
    ),
)


def parse_args() -> argparse.Namespace:
    labels = ", ".join(r.label for r in RESOURCES)
    parser = argparse.ArgumentParser(description="Download the Meno RAG stand resources from Yandex Disk.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources/stand_nsu"),
        help="Directory that should contain the corpus, mapping, abbreviations, and knowledge/ folder.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=f"Comma-separated subset of resources to fetch. Valid labels: {labels}.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Temporary zip path for folder resources (bm25). Defaults to <output-dir>/meno-rag-bm25.zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download resources even if their target files already exist.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the temporary zip archive(s) after successful extraction.",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve and print the temporary Yandex Disk download URLs.",
    )
    return parser.parse_args()


def split_public_url(public_url: str) -> tuple[str, str | None]:
    parsed = urllib.parse.urlparse(public_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Yandex Disk URL: {public_url!r}")

    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if "d" not in parts:
        return public_url, None

    disk_id_index = parts.index("d")
    if disk_id_index + 1 >= len(parts):
        return public_url, None

    public_key_path = "/" + "/".join(parts[: disk_id_index + 2])
    public_key = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, public_key_path, "", "", ""))
    resource_parts = parts[disk_id_index + 2 :]
    if not resource_parts:
        return public_key, None
    return public_key, "/" + "/".join(resource_parts)


def resolve_download_url(public_url: str, path: str | None = None) -> str:
    public_key, url_path = split_public_url(public_url)
    effective_path = path or url_path
    query = {"public_key": public_key}
    if effective_path:
        query["path"] = effective_path
    request_url = f"{DOWNLOAD_API_URL}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": "meno-rag-downloader/1.0"})

    try:
        with urllib.request.urlopen(request, timeout=60, context=ssl_context()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Yandex Disk API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        payload = curl_read(request_url)
        try:
            payload = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as json_exc:
            raise RuntimeError(f"Failed to reach Yandex Disk API: {exc}") from json_exc

    href = payload.get("href")
    if not href:
        raise RuntimeError(f"Yandex Disk API response does not contain a download URL: {payload!r}")
    return str(href)


def resource_expected_files(resource: Resource) -> list[str]:
    if resource.kind == "archive":
        return [f"{resource.target}/{name}" for name in BM25_FILES]
    return [resource.target]


def resource_satisfied(resource: Resource, output_dir: Path) -> bool:
    return all((output_dir / rel).is_file() for rel in resource_expected_files(resource))


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "meno-rag-downloader/1.0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60, context=ssl_context()) as response,
            dest.open("wb") as sink,
        ):
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_report = time.monotonic()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                sink.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    print_progress(downloaded, total)
                    last_report = now
            print_progress(downloaded, total)
            print(file=sys.stderr)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        print(f"Python download failed ({exc}); retrying with curl.", file=sys.stderr)
        curl_download(url, dest)


def download_resource(resource: Resource, output_dir: Path, archive_path: Path) -> None:
    href = resolve_download_url(resource.public_url, resource.path)
    if resource.kind == "file":
        target = output_dir / resource.target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        download_file(href, partial)
        partial.replace(target)
        return
    if resource.kind == "archive":
        download_file(href, archive_path)
        extract_bm25(archive_path, output_dir / resource.target)
        return
    raise RuntimeError(f"Unknown resource kind {resource.kind!r} for {resource.label!r}.")


def extract_bm25(archive_path: Path, target_dir: Path) -> None:
    """Extract the bm25 folder zip and place its files at ``target_dir``.

    The archive layout can vary (files at the root or under a ``bm25/`` folder),
    so we locate the directory that holds the bm25s marker file and move it into
    place, replacing any previous index.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        safe_extract(archive_path, tmp_dir)
        marker = next(tmp_dir.rglob("params.index.json"), None)
        if marker is None:
            raise RuntimeError(f"bm25 archive is missing params.index.json: {archive_path}")
        source_dir = marker.parent
        missing = [name for name in BM25_FILES if not (source_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"bm25 archive is incomplete; missing: {missing}")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_dir), str(target_dir))


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def curl_read(url: str) -> bytes:
    if shutil.which("curl") is None:
        raise RuntimeError("Failed to reach Yandex Disk API and curl is not available as a fallback.")
    completed = subprocess.run(
        ["curl", "-fsSL", url],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl failed while resolving Yandex Disk URL: {stderr}")
    return completed.stdout


def curl_download(url: str, archive_path: Path) -> None:
    if shutil.which("curl") is None:
        raise RuntimeError("Download failed and curl is not available as a fallback.")
    completed = subprocess.run(
        ["curl", "-fL", url, "-o", str(archive_path)],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"curl failed while downloading archive with exit code {completed.returncode}.")


def print_progress(downloaded: int, total: int) -> None:
    downloaded_mib = downloaded / 1024 / 1024
    if total > 0:
        total_mib = total / 1024 / 1024
        percent = downloaded / total * 100
        message = f"\rDownloaded {downloaded_mib:.1f}/{total_mib:.1f} MiB ({percent:.1f}%)"
    else:
        message = f"\rDownloaded {downloaded_mib:.1f} MiB"
    print(message, end="", file=sys.stderr, flush=True)


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_root / member.filename).resolve()
            if target != output_root and output_root not in target.parents:
                raise RuntimeError(f"Refusing to extract archive member outside output directory: {member.filename}")
        archive.extractall(output_root)


def validate_output(output_dir: Path, resources: tuple[Resource, ...]) -> None:
    missing = [rel for res in resources for rel in resource_expected_files(res) if not (output_dir / rel).is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(f"Downloaded knowledge directory is incomplete. Missing files:\n{formatted}")


def select_resources(only: str | None) -> tuple[Resource, ...]:
    if not only:
        return RESOURCES
    wanted = {token.strip() for token in only.split(",") if token.strip()}
    known = {res.label for res in RESOURCES}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"error: unknown --only labels {sorted(unknown)}; valid: {sorted(known)}")
    return tuple(res for res in RESOURCES if res.label in wanted)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    archive_path = (args.archive or output_dir / "meno-rag-bm25.zip").resolve()
    resources = select_resources(args.only)

    if args.resolve_only:
        for res in resources:
            print(f"{res.label}: {resolve_download_url(res.public_url, res.path)}", flush=True)
        return 0

    for res in resources:
        if resource_satisfied(res, output_dir) and not args.force:
            print(f"[skip] {res.label}: already present", flush=True)
            continue
        print(f"[get ] {res.label} -> {res.target}", flush=True)
        download_resource(res, output_dir, archive_path)

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)
    elif archive_path.is_file():
        print(f"Archive kept at {archive_path}", flush=True)

    validate_output(output_dir, resources)
    print(f"Stand resources are ready in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")
    except Exception as exc:
        raise SystemExit(f"error: {exc}")
