"""
Download mod/dat files for QPCC problems listed in macmpec_qpcc.json and convert
each to .nl using AMPL's `write g` command.

JSON keys are used directly as .nl file stems — guaranteed 1:1 correspondence.

Usage:
    python3 download_and_convert.py

Outputs:
    MacMPEC-problems/   — downloaded .mod and .dat files (cached across runs)
    MacMPEC-nl/         — one .nl file per problem, named by JSON key
"""

import subprocess, json
from pathlib import Path
from amplpy import AMPL

HERE         = Path(__file__).parent
PROBLEMS_DIR = HERE / "MacMPEC-problems"
NL_DIR       = HERE / "MacMPEC-nl"
JSON_PATH    = HERE / "macmpec_qpcc.json"
WIKI_BASE    = "https://wiki.mcs.anl.gov"
ANL_OLD      = "http://www.mcs.anl.gov"
ANL_NEW      = "https://www.mcs.anl.gov"

PROBLEMS_DIR.mkdir(exist_ok=True)
NL_DIR.mkdir(exist_ok=True)


def normalise_url(href: str) -> str:
    if href.startswith("/leyffer/"):
        return WIKI_BASE + href
    return href.replace(ANL_OLD, ANL_NEW)


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    result = subprocess.run(
        ["curl", "-sL", "--fail", "--max-time", "30", "-o", str(dest), url],
        capture_output=True,
    )
    if result.returncode == 0 and dest.exists():
        return True
    dest.unlink(missing_ok=True)
    print(f"    DOWNLOAD FAILED  {url}  (exit {result.returncode})")
    return False


def ampl_to_nl(problem_name: str, mod_path: Path, dat_path) -> bool:
    stub    = NL_DIR / problem_name
    nl_file = Path(str(stub) + ".nl")

    if nl_file.exists():
        return True

    ampl = AMPL()
    ampl.eval("option compl_warn 0;")
    ampl.eval(f'model "{mod_path.resolve()}";')
    if dat_path is not None:
        ampl.eval(f'data "{Path(dat_path).resolve()}";')
    ampl.eval(f'write "g{stub.resolve()}";')

    # Remove auxiliary files AMPL writes alongside .nl (not needed)
    for ext in (".row", ".col", ".fix", ".dat"):
        Path(str(stub) + ext).unlink(missing_ok=True)

    return nl_file.exists()


# ── Load QPCC problem list ─────────────────────────────────────────────────────

data = json.loads(JSON_PATH.read_text())
print(f"Loaded {len(data)} QPCC problems from {JSON_PATH.name}\n")

ok = skipped = failed = 0

for problem_name, entry in data.items():
    mod_href = entry.get("mod_file")
    dat_href = entry.get("dat_file")

    if not mod_href:
        print(f"  SKIP  {problem_name}  (no mod file)")
        skipped += 1
        continue

    mod_url = normalise_url(mod_href)
    dat_url = normalise_url(dat_href) if dat_href else None

    mod_path = PROBLEMS_DIR / Path(mod_url).name
    dat_path = PROBLEMS_DIR / Path(dat_url).name if dat_url else None

    if not download(mod_url, mod_path):
        failed += 1
        print(f"  FAIL  {problem_name}  (mod download failed)")
        continue
    if dat_url and dat_path is not None and not download(dat_url, dat_path):
        failed += 1
        print(f"  FAIL  {problem_name}  (dat download failed)")
        continue

    if ampl_to_nl(problem_name, mod_path, dat_path):
        print(f"  OK    {problem_name}.nl")
        ok += 1
    else:
        failed += 1
        print(f"  FAIL  {problem_name}  (AMPL conversion failed)")

print(f"\nDone: {ok} converted, {skipped} skipped, {failed} failed")
print(f"  .mod/.dat → {PROBLEMS_DIR}")
print(f"  .nl files → {NL_DIR}")
