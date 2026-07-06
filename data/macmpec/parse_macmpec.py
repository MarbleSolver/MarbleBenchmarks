import json, re
from pathlib import Path
from bs4 import BeautifulSoup

HERE = Path(__file__).parent

# Classification format: XXX-XX-CCC-n-m-p
#   Char 1: objective type       — C=constant, L=linear, Q=quadratic  (≤ quadratic)
#   Char 2: constraint type      — U=unconstrained, X=fixed, B=bounds, L=linear  (≤ linear)
#   CCC:    complementarity type — LCP, LMCP, or NLP (KKT of a program; may still be linear)
#
# NOTE: Q-objective problems are included but convex_quadratic is set to None — run
#       check_convexity.jl to fill those in by computing the objective Hessian from the .nl file.
def is_qpcc(cls: str) -> bool:
    parts = cls.split("-")
    if len(parts) < 3:
        return False
    obj_char  = parts[0][0]   # objective
    con_char  = parts[0][1]   # general constraints
    cc_type   = parts[2]      # complementarity type
    return (
        obj_char in "CLQ"
        and con_char in "UXBL"
        and cc_type in ("LCP", "NLP")
    )

soup = BeautifulSoup((HERE / "html" / "MacMPEC").read_text(), "html.parser")
table = soup.find_all("table")[1]   # second table is the problem list
rows  = table.find_all("tr")[1:]    # skip header row

result = {}

for row in rows:
    cells = row.find_all(["td", "th"])
    if len(cells) < 4:
        continue

    name           = cells[0].get_text(strip=True)
    classification = cells[3].get_text(strip=True)
    solution_text  = cells[4].get_text(strip=True) if len(cells) >= 5 else ""

    if not name or not classification:
        continue
    # if not is_qpcc(classification):
    #     continue

    mod_links = cells[1].find_all("a")
    dat_links = cells[2].find_all("a")

    mod_file = mod_links[0]["href"] if mod_links else cells[1].get_text(strip=True)
    dat_raw  = cells[2].get_text(strip=True)
    dat_file = dat_links[0]["href"] if dat_links else (None if dat_raw == "n/a" else dat_raw)

    try:
        solution = float(solution_text)
    except ValueError:
        solution = None

    mod_stem = Path(mod_file).stem if mod_file else name
    dat_stem = Path(dat_file).stem if dat_file else None
    problem_key = (f"{mod_stem}-{dat_stem}" if dat_stem else mod_stem).lower()

    obj_char = classification[0]
    result[problem_key] = {
        "display_name":     name,
        "classification":   classification,
        "mod_file":         mod_file,
        "dat_file":         dat_file,
        "solution":         solution,
        "convex_cost":      None,
        "linear_constraints": None,
    }

out = HERE / "macmpec_qpcc.json"
out.write_text(json.dumps(result, indent=2))

print(f"Found {len(result)} QPCC problems → {out}")
for key, v in sorted(result.items()):
    print(f"  {key:<30}  {v['classification']}")
