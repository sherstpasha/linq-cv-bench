import argparse
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_LINKS = [
    "https://disk.yandex.kz/d/UZKHkwPUpDW5iA",
    "https://disk.yandex.kz/d/c2RWNZ15QMR88g",
    "https://disk.yandex.kz/d/sCMhKf4lrC3CrA",
]
YANDEX_PUBLIC_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract archives from Yandex Disk public links")
    parser.add_argument("--links", nargs="+", default=DEFAULT_LINKS, help="Yandex Disk public links")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory for downloads and extraction")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded archives after extraction")
    parser.add_argument("--dry-run", action="store_true", help="Resolve links only, do not download")
    return parser.parse_args()


def get_download_href(public_link: str, timeout: int) -> str:
    query = urllib.parse.urlencode({"public_key": public_link})
    url = f"{YANDEX_PUBLIC_API}?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    href = payload.get("href")
    if not href:
        message = payload.get("message", "Unknown Yandex API error")
        raise RuntimeError(f"Failed to resolve link: {public_link} ({message})")
    return href


def infer_filename(href: str) -> str:
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)

    for key in ("filename", "name"):
        if key in qs and qs[key]:
            return qs[key][0]

    basename = Path(parsed.path).name
    if basename:
        return basename

    return "downloaded_file"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def download_file(href: str, out_path: Path, timeout: int) -> None:
    with urllib.request.urlopen(href, timeout=timeout) as resp, out_path.open("wb") as f:
        shutil.copyfileobj(resp, f)


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(
        name.endswith(ext)
        for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz")
    )


def unpack_archive(path: Path, target_dir: Path) -> Optional[Path]:
    if not is_archive(path):
        return None

    extract_dir = target_dir / path.stem
    if path.name.lower().endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz", ".txz")):
        # Remove only the last extension for tar-compressed names.
        extract_dir = target_dir / path.name.split(".")[0]

    extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(path.as_posix(), extract_dir.as_posix())
    flatten_duplicate_nested_dir(extract_dir)
    return extract_dir


def flatten_duplicate_nested_dir(extract_dir: Path) -> None:
    entries = list(extract_dir.iterdir())
    if len(entries) != 1:
        return

    nested = entries[0]
    if not nested.is_dir():
        return

    if nested.name.lower() != extract_dir.name.lower():
        return

    for item in nested.iterdir():
        destination = extract_dir / item.name
        if destination.exists():
            raise RuntimeError(f"Cannot flatten nested dir, target already exists: {destination}")
        shutil.move(item.as_posix(), destination.as_posix())
    nested.rmdir()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir: {args.data_dir}")

    for public_link in args.links:
        print(f"\nResolving: {public_link}")
        try:
            href = get_download_href(public_link, args.timeout)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        filename = infer_filename(href)
        out_path = unique_path(args.data_dir / filename)
        print(f"  File: {out_path.name}")

        if args.dry_run:
            print("  Dry run: skipping download")
            continue

        try:
            print("  Downloading...")
            download_file(href, out_path, args.timeout)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ERROR while downloading: {e}")
            continue

        try:
            extracted = unpack_archive(out_path, args.data_dir)
            if extracted is not None:
                print(f"  Extracted to: {extracted}")
                if not args.keep_archives:
                    out_path.unlink(missing_ok=True)
                    print("  Archive removed")
            else:
                print("  Not an archive, skipping extraction")
        except Exception as e:
            print(f"  ERROR while extracting: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
