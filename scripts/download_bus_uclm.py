"""Download and verify BUS-UCLM V3 from its official Mendeley file manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "BUS-UCLM"
SOURCES = {
    "images": ROOT / "data" / "bus_uclm_images_files.json",
    "masks": ROOT / "data" / "bus_uclm_masks_files.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(kind: str, record: dict) -> tuple[str, str]:
    destination = DATA_ROOT / kind / record["filename"]
    expected = record["content_details"]["sha256_hash"]
    if destination.is_file() and sha256(destination) == expected:
        return kind, record["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        [
            "curl.exe",
            "-fsSL",
            "--retry",
            "3",
            record["content_details"]["download_url"],
            "-o",
            str(partial),
        ],
        check=True,
        timeout=180,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    actual = sha256(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for {kind}/{record['filename']}: {actual}")
    partial.replace(destination)
    return kind, record["filename"]


def main() -> None:
    tasks = []
    for kind, manifest_path in SOURCES.items():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks.extend((kind, record) for record in records)
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_one, *task) for task in tasks]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(
                    f"[BUS-UCLM] verified {completed}/{len(tasks)} files",
                    flush=True,
                )
    print(f"[DONE] {DATA_ROOT}")


if __name__ == "__main__":
    main()
