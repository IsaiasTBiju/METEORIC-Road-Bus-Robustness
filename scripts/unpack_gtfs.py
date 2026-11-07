"""Unpack the RTA GTFS archive (.7z containing a .zip) into data/interim/gtfs/.

Re-runnable: skips the work if already done, unless --force is given.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import py7zr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gtfs"
OUT = ROOT / "data" / "interim" / "gtfs"


def main(force: bool) -> None:
    archives = sorted(RAW.glob("*.7z"))
    if len(archives) != 1:
        raise SystemExit(f"Expected exactly one .7z in {RAW}, found {len(archives)}")

    feed_dir = OUT / "feed"
    if feed_dir.exists() and not force:
        print(f"Already unpacked at {feed_dir} (use --force to redo)")
        return
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    with py7zr.SevenZipFile(archives[0]) as archive:
        archive.extractall(path=OUT)

    zips = sorted(OUT.glob("*.zip"))
    if len(zips) != 1:
        raise SystemExit(f"Expected one .zip inside the archive, found {len(zips)}")
    with zipfile.ZipFile(zips[0]) as feed:
        feed.extractall(feed_dir)

    print(f"Unpacked {archives[0].name} -> {feed_dir}")
    for f in sorted(feed_dir.iterdir()):
        print(f"  {f.name:22s} {f.stat().st_size:>13,d} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redo even if already unpacked")
    main(parser.parse_args().force)
