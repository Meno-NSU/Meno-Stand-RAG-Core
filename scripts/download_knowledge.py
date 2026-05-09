#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_PUBLIC_URL = "https://disk.yandex.ru/d/eklv6Scj9OpbmQ/knowledge"
DOWNLOAD_API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
EXPECTED_FILES = (
    "knowledge/faiss_frida.index",
    "knowledge/bm25/data.csc.index.npy",
    "knowledge/bm25/indices.csc.index.npy",
    "knowledge/bm25/indptr.csc.index.npy",
    "knowledge/bm25/params.index.json",
    "knowledge/bm25/vocab.index.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Meno RAG knowledge resources from Yandex Disk.")
    parser.add_argument(
        "--url",
        default=DEFAULT_PUBLIC_URL,
        help=f"Yandex Disk public URL to the knowledge folder. Defaults to {DEFAULT_PUBLIC_URL!r}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources/stand_nsu"),
        help="Directory that should contain the extracted knowledge/ folder.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Temporary zip archive path. Defaults to <output-dir>/knowledge.zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and extract even if the expected knowledge files already exist.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip archive after successful extraction.",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve and print the temporary Yandex Disk download URL.",
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


def resolve_download_url(public_url: str) -> str:
    public_key, resource_path = split_public_url(public_url)
    query = {"public_key": public_key}
    if resource_path:
        query["path"] = resource_path
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


def expected_files_exist(output_dir: Path) -> bool:
    return all((output_dir / relative_path).is_file() for relative_path in EXPECTED_FILES)


def download_file(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "meno-rag-downloader/1.0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60, context=ssl_context()) as response,
            archive_path.open("wb") as archive,
        ):
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_report = time.monotonic()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                archive.write(chunk)
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
        curl_download(url, archive_path)


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


def normalize_extracted_layout(output_dir: Path) -> None:
    if (output_dir / "knowledge").is_dir():
        return

    root_faiss = output_dir / "faiss_frida.index"
    root_bm25 = output_dir / "bm25"
    if not root_faiss.is_file() or not root_bm25.is_dir():
        return

    knowledge_dir = output_dir / "knowledge"
    knowledge_dir.mkdir()
    shutil.move(str(root_faiss), knowledge_dir / root_faiss.name)
    shutil.move(str(root_bm25), knowledge_dir / root_bm25.name)


def validate_output(output_dir: Path) -> None:
    missing = [relative_path for relative_path in EXPECTED_FILES if not (output_dir / relative_path).is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(f"Downloaded knowledge directory is incomplete. Missing files:\n{formatted}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    archive_path = (args.archive or output_dir / "knowledge.zip").resolve()

    if expected_files_exist(output_dir) and not args.force and not args.resolve_only:
        print(f"Knowledge resources already exist in {output_dir / 'knowledge'}", flush=True)
        return 0

    print(f"Resolving Yandex Disk download URL for {args.url}", flush=True)
    download_url = resolve_download_url(args.url)
    if args.resolve_only:
        print(download_url, flush=True)
        return 0

    print(f"Downloading archive to {archive_path}", flush=True)
    download_file(download_url, archive_path)

    print(f"Extracting archive into {output_dir}", flush=True)
    safe_extract(archive_path, output_dir)
    normalize_extracted_layout(output_dir)
    validate_output(output_dir)

    if args.keep_archive:
        print(f"Archive kept at {archive_path}", flush=True)
    else:
        archive_path.unlink(missing_ok=True)

    print(f"Knowledge resources are ready in {output_dir / 'knowledge'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")
    except Exception as exc:
        raise SystemExit(f"error: {exc}")
