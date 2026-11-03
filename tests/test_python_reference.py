"""Execute the reference's Python blocks against the matching built extension."""
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.environ["PYNEURONET_BUILD_DIR"])
reference = Path(sys.argv[1])
blocks = re.findall(r"^```python\n(.*?)^```", reference.read_text(encoding="utf-8"), re.M | re.S)
if not blocks:
    raise RuntimeError("No Python examples found in binding reference")
# compile(..., optimize=0) preserves documented assertions even under python -O.
namespace = {"__name__": "__main__"}
for index, block in enumerate(blocks, 1):
    exec(compile(block, f"{reference}:block{index}", "exec", optimize=0), namespace)
print(f"Passed {len(blocks)} Python reference block(s)")
