from pathlib import Path
from amplpy import AMPL

HERE = Path(__file__).parent
mod  = HERE / "MacMPEC-problems" / "bard2.mod"
out  = HERE / "bard2_regen"

ampl = AMPL()
ampl.eval("option compl_warn 0;")
ampl.eval(f'model "{mod.resolve()}";')
ampl.eval(f'write "g{out.resolve()}";')

for ext in (".row", ".col", ".fix", ".dat"):
    Path(str(out) + ext).unlink(missing_ok=True)

print(f"Wrote {out}.nl")
