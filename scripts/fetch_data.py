"""Download the archived data of the paper from Zenodo and unpack it into results/.

The Zenodo record (data_sources.yaml) stores results/ in parts: data_{problem}.tar (raw run
data of one problem), fronts.tar (reference and pooled fronts) and derived.tar.gz (thresholds,
references, summaries, tables, figures, timing), next to README.md and SHA256SUMS.

    uv run python scripts/fetch_data.py derived              # enough to rebuild tables and figures
    uv run python scripts/fetch_data.py raw --problem g9     # run data of some problems
    uv run python scripts/fetch_data.py fronts
    uv run python scripts/fetch_data.py all                  # everything, about 17 GB

Each file is downloaded into .downloads/ (resumed if interrupted), checked against the MD5 in
data_sources.yaml, unpacked into results/ with its original modification times, and every
unpacked file is checked against SHA256SUMS. Finally a Snakemake dry run shows what the pipeline
would still do; it warns if that includes rerunning experiments.

Options: --token for a Zenodo share link of an unpublished record, --from-dir to use files
downloaded by hand instead of downloading, --results to unpack elsewhere, --keep to keep the
downloaded archives, --no-dry-run to skip the Snakemake check.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 * 1024 * 1024


def progress(done: int, total: int, note: str) -> None:
    print(f"##slip-progress {done}/{total} {note}", flush=True)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, tries: int = 5) -> None:
    """Download url to dest, resuming a partial file; retries on network errors."""
    for attempt in range(1, tries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return
        if have > size:
            dest.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"User-Agent": "shadow-price-experiments fetch_data.py"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the range: start over
                    have = 0
                mode = "ab" if have else "wb"
                with dest.open(mode) as f:
                    done, last = have, 0.0
                    for b in iter(lambda: r.read(CHUNK), b""):
                        f.write(b)
                        done += len(b)
                        if time.time() - last > 2:
                            print(f"  {dest.name}: {done / 1e9:.2f} of {size / 1e9:.2f} GB", flush=True)
                            last = time.time()
            if dest.stat().st_size == size:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  {dest.name}: attempt {attempt} failed ({e}); retrying", flush=True)
            time.sleep(min(60, 5 * attempt))
    sys.exit(f"{dest.name}: download incomplete after {tries} attempts; run again to resume")


def safe_extract(archive: Path, target: Path) -> list[str]:
    """Unpack archive into target (relative paths only, mtimes kept); return the member names."""
    names = []
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                sys.exit(f"{archive.name}: unsafe member path {m.name}")
        tf.extractall(target, members=members, filter="data")
        names = [m.name for m in members]
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["derived", "raw", "fronts", "all"])
    ap.add_argument("--problem", nargs="*", help="with raw: only these problems (default: all)")
    ap.add_argument("--token", help="access token of a Zenodo share link (unpublished record)")
    ap.add_argument("--from-dir", type=Path, help="use files already downloaded into this directory")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--downloads", type=Path, default=ROOT / ".downloads")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded archives")
    ap.add_argument("--no-dry-run", action="store_true")
    args = ap.parse_args()

    src = yaml.safe_load((ROOT / "data_sources.yaml").read_text())
    record = src["zenodo"]["record"]
    files = {f["name"]: f for f in src["files"]}

    parts = {"derived": ["derived"], "raw": ["raw"], "fronts": ["fronts"], "all": ["derived", "raw", "fronts"]}[args.what]
    wanted = [f for f in files.values() if f["part"] in parts]
    if args.problem:
        known = {f["name"][len("data_"):-len(".tar")] for f in files.values() if f["part"] == "raw"}
        unknown = set(args.problem) - known
        if unknown:
            sys.exit(f"unknown problem(s) {sorted(unknown)}; known: {sorted(known)}")
        wanted = [f for f in wanted if f["part"] != "raw" or f["name"][len("data_"):-len(".tar")] in args.problem]
    wanted.append(files["SHA256SUMS"])

    def url(name: str) -> str:
        u = f"https://zenodo.org/records/{record}/files/{name}?download=1"
        return u + (f"&token={args.token}" if args.token else "")

    dl = args.from_dir or args.downloads
    dl.mkdir(parents=True, exist_ok=True)
    total = sum(f["size"] for f in wanted)
    print(f"{len(wanted)} files, {total / 1e9:.2f} GB, from "
          + (str(args.from_dir) if args.from_dir else f"Zenodo record {record}"), flush=True)

    done = 0
    for f in wanted:
        path = dl / f["name"]
        if not args.from_dir:
            download(url(f["name"]), path, f["size"])
        if not path.exists():
            sys.exit(f"{path} is missing")
        if path.stat().st_size != f["size"] or md5sum(path) != f["md5"]:
            sys.exit(f"{path.name}: size or MD5 differs from data_sources.yaml; delete it and run again")
        done += f["size"]
        progress(done * 500 // total, 1000, f"{f['name']} downloaded and checked")

    sums = {}
    for line in (dl / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        sums[name] = h

    args.results.mkdir(parents=True, exist_ok=True)
    archives = [f for f in wanted if f["name"].endswith((".tar", ".tar.gz"))]
    bad, n_checked = [], 0
    for i, f in enumerate(archives, 1):
        progress(500 + (i - 1) * 500 // len(archives), 1000, f"unpacking {f['name']}")
        for name in safe_extract(dl / f["name"], args.results):
            h = hashlib.sha256()
            with (args.results / name).open("rb") as fh:
                for b in iter(lambda: fh.read(CHUNK), b""):
                    h.update(b)
            n_checked += 1
            if sums.get(name) != h.hexdigest():
                bad.append(name)
        if not args.keep and not args.from_dir:
            (dl / f["name"]).unlink()
    progress(1000, 1000, "unpacked and checked")
    if bad:
        sys.exit(f"{len(bad)} unpacked files differ from SHA256SUMS, e.g. {bad[:5]}")
    print(f"unpacked into {args.results}: {n_checked} files, all match SHA256SUMS", flush=True)

    if args.no_dry_run:
        return
    cmd = ["uv", "run", "snakemake", "-n", "--quiet", "rules", "--config", f"results_dir={args.results}"]
    if args.what == "derived":
        cmd += ["from_summaries=true"]
    print("Snakemake dry run:", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(out.stdout[-3000:] or out.stderr[-3000:])
    if re.search(r"^\s*(generate_data|generate_pareto_front)\s+\d+\s*$", out.stdout, re.M):
        print("WARNING: the pipeline would rerun experiments (generate_data or generate_pareto_front). "
              "That takes a long time; fetch the missing parts first (e.g. 'all'), or run with "
              "--config from_summaries=true to rebuild tables and figures from the derived data only.")


if __name__ == "__main__":
    main()
