"""Download TCPD, SKAB (and optionally TCPDBench) into data/raw/ and build the 10 TCPD series
that are fetched from external sources. Works on Windows, macOS and Linux. Needs `git` and internet.

    python scripts/download_data.py                 # TCPD + SKAB (needed to run the project)
    python scripts/download_data.py --tcpdbench     # also TCPDBench results (~310 MB, only for validation tests)

TCPD's own build script needs extra packages (yfinance, clevercsv, ...). They are installed into a
separate virtual environment, data/raw/.tcpd_env, so the project environment stays clean.
"""
import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "TCPD": "https://github.com/alan-turing-institute/TCPD.git",
    "SKAB": "https://github.com/waico/SKAB.git",
}
TCPDBENCH = "https://github.com/alan-turing-institute/TCPDBench.git"
TCPDBENCH_PATHS = ["analysis/scripts", "analysis/annotations", "analysis/output/scores",
                   "analysis/output/tables", "analysis/output/summaries"]

# Runs TCPD's build_tcpd.py with Unix newlines for every text file it writes: on Windows the default
# "\r\n" changes the files' MD5 and TCPD's checksum validation fails.
LF_RUNNER = """
import builtins, runpy, sys
_open = builtins.open
def _open_lf(file, mode="r", *a, **kw):
    if ("w" in mode or "a" in mode) and "b" not in mode and "newline" not in kw and len(a) < 3:
        kw["newline"] = "\\n"
    return _open(file, mode, *a, **kw)
builtins.open = _open_lf
sys.argv = ["build_tcpd.py", "-v", "collect"]
runpy.run_path("build_tcpd.py", run_name="__main__")
"""


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def clone(url, dest, sparse_paths=None):
    if dest.exists():
        print(f"= {dest.name} already present, skipping clone")
        return
    if sparse_paths:
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, dest])
        run(["git", "sparse-checkout", "set", *sparse_paths], cwd=dest)
    else:
        run(["git", "clone", "--depth", "1", url, dest])


def build_tcpd(raw):
    env = raw / ".tcpd_env"
    py = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not py.exists():
        venv.create(env, with_pip=True)
        run([py, "-m", "pip", "install", "-q", "-r", raw / "TCPD" / "requirements.txt"])
    runner = raw / ".tcpd_env" / "run_build_lf.py"
    runner.write_text(LF_RUNNER)
    run([py, runner], cwd=raw / "TCPD")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcpdbench", action="store_true", help="also fetch TCPDBench results (validation tests)")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"), help="destination (default: data/raw)")
    args = ap.parse_args()
    raw = Path(args.raw_dir).resolve()  # absolute: the TCPD build runs with cwd inside raw/TCPD
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        clone(url, raw / name)
    build_tcpd(raw)
    if args.tcpdbench:
        clone(TCPDBENCH, raw / "TCPDBench", TCPDBENCH_PATHS)
    n_tcpd = len(list((raw / "TCPD" / "datasets").glob("*/*.json")))
    n_skab = len(list((raw / "SKAB" / "data").glob("*/*.csv")))
    print(f"\nDone: {n_tcpd}/42 TCPD series, {n_skab}/35 SKAB files in {raw}")
    if n_tcpd < 42:
        sys.exit("Some TCPD series failed to build (usually a temporary yfinance/archive.org outage): re-run.")


if __name__ == "__main__":
    main()
