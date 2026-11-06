"""Execute the reference's Python blocks against the matching built extension."""
from importlib.machinery import EXTENSION_SUFFIXES, PathFinder
import os
from pathlib import Path
import re
import sys

build_dir = Path(os.environ["PYNEURONET_BUILD_DIR"]).resolve()
spec = PathFinder.find_spec("pyneuronet", [str(build_dir)])
if (spec is None or spec.origin is None
        or Path(spec.origin).resolve().parent != build_dir
        or not any(spec.origin.endswith(suffix) for suffix in EXTENSION_SUFFIXES)):
    raise RuntimeError("Matching build-tree pyneuronet extension is required")
sys.path.insert(0, str(build_dir))
reference = Path(sys.argv[1])
blocks = re.findall(r"^```python\n(.*?)^```", reference.read_text(encoding="utf-8"), re.M | re.S)
if not blocks:
    raise RuntimeError("No Python examples found in binding reference")
# compile(..., optimize=0) preserves documented assertions even under python -O.
namespace = {"__name__": "__main__"}
for index, block in enumerate(blocks, 1):
    exec(compile(block, f"{reference}:block{index}", "exec", optimize=0), namespace)
print(f"Passed {len(blocks)} Python reference block(s)")
