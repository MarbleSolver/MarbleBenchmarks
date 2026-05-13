import subprocess, shutil
from pathlib import Path
from bs4 import BeautifulSoup
from amplpy import AMPL
from pathlib import Path
from amplpy import AMPL

NL_DIR = Path("/Users/micahreich/Documents/MarbleBenchmarks/data/macmpec/MacMPEC-nl")

def ampl_to_nl(problem_name: str, mod_path: Path, dat_path=None) -> bool:
    NL_DIR.mkdir(parents=True, exist_ok=True)

    stub = NL_DIR / problem_name
    nl_file = stub.with_suffix(".nl")
    if nl_file.exists():
        return True

    ampl = AMPL()

    # Suppress AMPL warning about nonsquare complementarity systems.
    ampl.eval("option compl_warn 0;")
    # or:
    # ampl.set_option("compl_warn", 0)

    ampl.eval(f'model "{mod_path.resolve()}";')
    if dat_path is not None:
        ampl.eval(f'data "{Path(dat_path).resolve()}";')

    ampl.eval(f'write g"{stub.resolve()}";')
    return nl_file.exists()

ampl_to_nl("ex9.2.1", Path("/Users/micahreich/Documents/MarbleBenchmarks/data/macmpec/MacMPEC-problems/ex9.2.1.mod"), None)